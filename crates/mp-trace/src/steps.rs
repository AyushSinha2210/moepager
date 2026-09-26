//! Grouping expert events into decode steps, and token-boundary inference
//! for black-box traces where the token index is unknown.

use std::collections::BTreeMap;

use crate::{ExpertEvent, TraceHeader};

/// Routing observed in one layer during one step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerUse {
    pub layer: u16,
    /// Sorted, distinct.
    pub experts: Vec<u16>,
    /// Time of the first event for this layer in this step.
    pub t_ns: u64,
}

/// One decode step (token).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub token: u32,
    /// Sorted by layer.
    pub layers: Vec<LayerUse>,
}

impl Step {
    pub fn t_ns(&self) -> u64 {
        self.layers.iter().map(|l| l.t_ns).min().unwrap_or(0)
    }
}

/// A trace as an ordered list of steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Steps {
    pub n_layers: u32,
    pub n_experts: u32,
    pub steps: Vec<Step>,
}

impl Steps {
    /// Group events by token, then layer; duplicate experts are removed.
    pub fn from_events(h: &TraceHeader, ev: &[ExpertEvent]) -> Self {
        let mut by: BTreeMap<u32, BTreeMap<u16, (Vec<u16>, u64)>> = BTreeMap::new();
        for e in ev {
            let slot = by
                .entry(e.token)
                .or_default()
                .entry(e.layer)
                .or_insert_with(|| (Vec::new(), e.t_ns));
            slot.0.push(e.expert);
            slot.1 = slot.1.min(e.t_ns);
        }
        let steps = by
            .into_iter()
            .map(|(token, layers)| Step {
                token,
                layers: layers
                    .into_iter()
                    .map(|(layer, (mut experts, t_ns))| {
                        experts.sort_unstable();
                        experts.dedup();
                        LayerUse {
                            layer,
                            experts,
                            t_ns,
                        }
                    })
                    .collect(),
            })
            .collect();
        Steps {
            n_layers: h.n_layers,
            n_experts: h.n_experts,
            steps,
        }
    }

    /// Flatten back to events in (token, layer, expert) order.
    pub fn to_events(&self) -> Vec<ExpertEvent> {
        let mut v = Vec::new();
        for s in &self.steps {
            for l in &s.layers {
                for &e in &l.experts {
                    v.push(ExpertEvent {
                        t_ns: l.t_ns,
                        token: s.token,
                        layer: l.layer,
                        expert: e,
                    });
                }
            }
        }
        v
    }

    /// Dense unit id used throughout the simulator: `layer * n_experts + expert`.
    pub fn unit(&self, layer: u16, expert: u16) -> u32 {
        layer as u32 * self.n_experts + expert as u32
    }

    pub fn n_units(&self) -> u32 {
        self.n_layers * self.n_experts
    }

    /// Total number of (step, layer, expert) accesses.
    pub fn n_accesses(&self) -> usize {
        self.steps
            .iter()
            .flat_map(|s| &s.layers)
            .map(|l| l.experts.len())
            .sum()
    }
}

/// Assign token indices to time-ordered events whose token is unknown.
///
/// Within one decode step, layers are visited in increasing order. A new
/// step starts when an event's layer is more than `slack` below the highest
/// layer seen in the current step. `slack` absorbs reordering from polling
/// (several layers discovered in one sweep). Returns the number of steps.
pub fn infer_tokens(ev: &mut [ExpertEvent], slack: u16) -> u32 {
    ev.sort_by_key(|e| e.t_ns);
    let mut token = 0u32;
    let mut max_layer: Option<u16> = None;
    for e in ev.iter_mut() {
        if let Some(m) = max_layer {
            if e.layer.saturating_add(slack) < m {
                token += 1;
                max_layer = None;
            }
        }
        max_layer = Some(max_layer.map_or(e.layer, |m| m.max(e.layer)));
        e.token = token;
    }
    if ev.is_empty() {
        0
    } else {
        token + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(t: u64, token: u32, layer: u16, expert: u16) -> ExpertEvent {
        ExpertEvent {
            t_ns: t,
            token,
            layer,
            expert,
        }
    }

    #[test]
    fn groups_and_dedups() {
        let h = TraceHeader::new("t", 2, 4);
        let e = [
            ev(5, 0, 1, 3),
            ev(1, 0, 0, 2),
            ev(2, 0, 0, 2),
            ev(3, 0, 0, 1),
            ev(9, 1, 0, 0),
        ];
        let s = Steps::from_events(&h, &e);
        assert_eq!(s.steps.len(), 2);
        assert_eq!(
            s.steps[0].layers[0],
            LayerUse {
                layer: 0,
                experts: vec![1, 2],
                t_ns: 1
            }
        );
        assert_eq!(s.steps[0].layers[1].experts, vec![3]);
        assert_eq!(s.n_accesses(), 4);
        assert_eq!(s.unit(1, 3), 7);
        let back = Steps::from_events(&h, &s.to_events());
        assert_eq!(back, s);
    }

    #[test]
    fn infers_token_boundaries() {
        // Two tokens over 6 layers; layer 2 discovered slightly late in token 0.
        let mut e: Vec<ExpertEvent> = [0u16, 1, 3, 2, 4, 5, 0, 1, 2, 3, 4, 5]
            .iter()
            .enumerate()
            .map(|(i, &l)| ev(i as u64, 99, l, 0))
            .collect();
        assert_eq!(infer_tokens(&mut e, 1), 2);
        assert!(e[..6].iter().all(|x| x.token == 0));
        assert!(e[6..].iter().all(|x| x.token == 1));
        assert_eq!(infer_tokens(&mut [], 1), 0);
    }
}
