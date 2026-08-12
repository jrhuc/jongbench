use super::EventWithCanAct;
use super::{Event, EventExt};
use crate::agent::{BatchAgent, MortalBatchAgent};
use crate::state::PlayerState;

use anyhow::{Context, Result, ensure};
use pyo3::prelude::*;
use serde_json as json;

#[pyclass]
pub struct Bot {
    agent: MortalBatchAgent,
    state: PlayerState,
    log: Vec<EventExt>,
}

#[pymethods]
impl Bot {
    #[new]
    fn new(engine: PyObject, player_id: u8) -> Result<Self> {
        let agent = MortalBatchAgent::new(engine, &[player_id])?;
        let state = PlayerState::new(player_id);
        Ok(Self {
            agent,
            state,
            log: vec![],
        })
    }

    /// Returns the reaction to `line`, if it can react, `None` otherwise.
    ///
    /// Set `can_act` or `line_json['can_act']` to `False` to force the bot to
    /// only update its state without making any reaction.
    ///
    /// Both `line` and the return value are JSON strings representing one
    /// single mjai event.
    #[pyo3(name = "react")]
    #[pyo3(signature = (line, /, *, can_act=true))]
    fn react_py(&mut self, line: &str, can_act: bool, py: Python<'_>) -> Result<Option<String>> {
        py.allow_threads(move || self.react(line, can_act))
    }
}

impl Bot {
    fn react(&mut self, line: &str, can_act: bool) -> Result<Option<String>> {
        let data: EventWithCanAct =
            json::from_str(line).with_context(|| format!("failed to parse event {line}"))?;

        match data.event {
            Event::StartGame { .. } => {
                self.agent.start_game(0)?;
            }
            Event::EndKyoku => {
                self.log.clear();
                self.agent.end_kyoku(0)?;
            }
            Event::EndGame => {
                self.agent.end_game(0, &Default::default())?;
            }
            _ => {
                self.log.push(EventExt::no_meta(data.event.clone()));
            }
        };

        let cans = self.state.update(&data.event)?;
        if !can_act || matches!(data.can_act, Some(false)) || !cans.can_act() {
            return Ok(None);
        }

        self.agent
            .set_scene(0, &self.log, &self.state, None)
            .context("failed to add state")?;
        let reaction = self
            .agent
            .get_reaction(0, &self.log, &self.state, None)
            .context("failed to get reaction")?;

        let ret = json::to_string(&reaction)?;
        Ok(Some(ret))
    }
}

#[pyclass]
pub struct BatchBot {
    agent: MortalBatchAgent,
    states: Vec<PlayerState>,
    logs: Vec<Vec<EventExt>>,
}

#[pymethods]
impl BatchBot {
    #[new]
    fn new(engine: PyObject, player_ids: Vec<u8>) -> Result<Self> {
        ensure!(
            matches!(player_ids.len(), 1..=4),
            "player_ids must contain between 1 and 4 seats"
        );

        let mut seen = [false; 4];
        for &player_id in &player_ids {
            ensure!(matches!(player_id, 0..=3), "invalid player id {player_id}");
            ensure!(
                !seen[usize::from(player_id)],
                "duplicate player id {player_id}"
            );
            seen[usize::from(player_id)] = true;
        }

        let agent = MortalBatchAgent::new(engine, &player_ids)?;
        let states: Vec<_> = player_ids.into_iter().map(PlayerState::new).collect();
        let logs = vec![vec![]; states.len()];
        Ok(Self {
            agent,
            states,
            logs,
        })
    }

    #[pyo3(name = "react")]
    #[pyo3(signature = (line, /, *, can_act=true))]
    fn react_py(
        &mut self,
        line: &str,
        can_act: bool,
        py: Python<'_>,
    ) -> Result<Vec<Option<String>>> {
        py.allow_threads(move || self.react(line, can_act))
    }
}

impl BatchBot {
    fn react(&mut self, line: &str, can_act: bool) -> Result<Vec<Option<String>>> {
        let data: EventWithCanAct =
            json::from_str(line).with_context(|| format!("failed to parse event {line}"))?;

        match data.event {
            Event::StartGame { .. } => {
                for index in 0..self.states.len() {
                    self.agent.start_game(index)?;
                }
            }
            Event::EndKyoku => {
                for log in &mut self.logs {
                    log.clear();
                }
                for index in 0..self.states.len() {
                    self.agent.end_kyoku(index)?;
                }
            }
            Event::EndGame => {
                for index in 0..self.states.len() {
                    self.agent.end_game(index, &Default::default())?;
                }
            }
            _ => {
                for log in &mut self.logs {
                    log.push(EventExt::no_meta(data.event.clone()));
                }
            }
        }

        let reactions_enabled = can_act && !matches!(data.can_act, Some(false));
        let mut active = Vec::with_capacity(self.states.len());
        for (index, state) in self.states.iter_mut().enumerate() {
            let cans = state.update(&data.event)?;
            if reactions_enabled && cans.can_act() {
                active.push(index);
            }
        }

        for &index in &active {
            self.agent
                .set_scene(index, &self.logs[index], &self.states[index], None)
                .context("failed to add state")?;
        }

        let mut reactions = vec![None; self.states.len()];
        for index in active {
            let reaction = self
                .agent
                .get_reaction(index, &self.logs[index], &self.states[index], None)
                .context("failed to get reaction")?;
            reactions[index] = Some(json::to_string(&reaction)?);
        }
        Ok(reactions)
    }
}
