//! /proc/<pid>/maps parsing, used to find where the engine mapped the
//! model file (needed for `process_madvise` demotion).

use std::io;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapEntry {
    pub start: u64,
    pub end: u64,
    pub perms: String,
    /// File offset of `start`.
    pub offset: u64,
    pub dev_major: u32,
    pub dev_minor: u32,
    pub inode: u64,
    pub path: Option<String>,
}

impl MapEntry {
    /// Virtual address of file offset `off`, if this mapping covers it.
    pub fn addr_of(&self, off: u64) -> Option<u64> {
        let len = self.end - self.start;
        (off >= self.offset && off < self.offset + len).then(|| self.start + (off - self.offset))
    }
}

/// Parse one line: `start-end perms offset maj:min inode [path]`.
pub fn parse_maps_line(line: &str) -> Option<MapEntry> {
    let mut it = line.split_whitespace();
    let (s, e) = it.next()?.split_once('-')?;
    let perms = it.next()?.to_string();
    let offset = u64::from_str_radix(it.next()?, 16).ok()?;
    let (maj, min) = it.next()?.split_once(':')?;
    let inode = it.next()?.parse().ok()?;
    let rest: Vec<&str> = it.collect();
    Some(MapEntry {
        start: u64::from_str_radix(s, 16).ok()?,
        end: u64::from_str_radix(e, 16).ok()?,
        perms,
        offset,
        dev_major: u32::from_str_radix(maj, 16).ok()?,
        dev_minor: u32::from_str_radix(min, 16).ok()?,
        inode,
        path: (!rest.is_empty()).then(|| rest.join(" ")),
    })
}

/// Mappings of a file in process `pid`, matched by inode plus either the
/// device or the path.
///
/// The device alone is not reliable: on btrfs `stat()` reports the
/// subvolume's anonymous device (e.g. 0:45) while /proc/pid/maps and the
/// filemap tracepoints use the superblock device (e.g. 0:23).
pub fn find_file_mapping(
    pid: u32,
    dev: (u32, u32),
    inode: u64,
    path: Option<&Path>,
) -> io::Result<Vec<MapEntry>> {
    let text = std::fs::read_to_string(format!("/proc/{pid}/maps"))?;
    let canon = path.and_then(|p| p.canonicalize().ok());
    Ok(text
        .lines()
        .filter_map(parse_maps_line)
        .filter(|m| {
            m.inode == inode
                && ((m.dev_major, m.dev_minor) == dev
                    || canon
                        .as_deref()
                        .is_some_and(|c| m.path.as_deref() == c.to_str()))
        })
        .collect())
}

/// The kernel's (superblock) device for a file this process has mapped,
/// as the filemap tracepoints report it in `s_dev`.
pub fn kernel_dev_of_mapping(inode: u64, path: &Path) -> io::Result<Option<(u32, u32)>> {
    let found = find_file_mapping(std::process::id(), (u32::MAX, u32::MAX), inode, Some(path))?;
    Ok(found.first().map(|m| (m.dev_major, m.dev_minor)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lines() {
        let l = "7f1c2a000000-7f1c6b000000 r--s 00001000 00:2a 123456   /models/Qwen3 30B.gguf";
        let m = parse_maps_line(l).unwrap();
        assert_eq!(m.start, 0x7f1c2a000000);
        assert_eq!(m.offset, 0x1000);
        assert_eq!((m.dev_major, m.dev_minor), (0, 0x2a));
        assert_eq!(m.inode, 123456);
        assert_eq!(m.path.as_deref(), Some("/models/Qwen3 30B.gguf"));
        assert_eq!(m.addr_of(0x1000), Some(0x7f1c2a000000));
        assert_eq!(m.addr_of(0x0), None);
        let anon = parse_maps_line("7ffd1000-7ffd2000 rw-p 00000000 00:00 0").unwrap();
        assert_eq!(anon.path, None);
        assert!(parse_maps_line("garbage").is_none());
    }
}
