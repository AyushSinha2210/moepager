# Ollama and Headless Engine Integration Considerations

Architecture considerations for using `moepager` with containerized or headless inference engines like Ollama.

## Transparent Interception
Because `moepager` operates purely at the OS page cache layer, any inference engine that maps GGUF files with `mmap(MAP_SHARED)` and avoids repacking will benefit from expert completion and pinning without needing API integration or IPC.

## Verifying Memory Sharing
Ensure the server process opens the file in `MAP_SHARED` mode. Private mappings (`MAP_PRIVATE`) create copy-on-write anonymous pages which prevent shared folio pinning.
