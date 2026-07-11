use super::game::{BatchGame, Index};
use crate::agent::{BatchAgent, new_py_agent};
use std::array;
use std::fs::{self, File};
use std::io;
use std::path::PathBuf;

use anyhow::{Result, anyhow, ensure};
use flate2::Compression;
use flate2::read::GzEncoder;
use pyo3::prelude::*;

/// An arena where four distinct engines play against each other, one engine
/// per seat. Engine `i` sits at seat `(i + g) % 4` in game `g`, so seats
/// rotate across games.
#[pyclass]
#[derive(Clone, Default)]
pub struct FourEngines {
    pub disable_progress_bar: bool,
    pub log_dir: Option<String>,
}

#[pymethods]
impl FourEngines {
    #[new]
    #[pyo3(signature = (*, disable_progress_bar=false, log_dir=None))]
    const fn new(disable_progress_bar: bool, log_dir: Option<String>) -> Self {
        Self {
            disable_progress_bar,
            log_dir,
        }
    }

    /// Plays `seed_count` hanchans among four engines, with seat rotation
    /// across games. Returns a list of `(names, scores, seed)` per game,
    /// where `names` and `scores` are in seat order (East first at start).
    pub fn py_4p(
        &self,
        engines: Vec<PyObject>,
        seed_start: (u64, u64),
        seed_count: u64,
        py: Python<'_>,
    ) -> Result<Vec<([String; 4], [i32; 4], (u64, u64))>> {
        ensure!(engines.len() == 4, "expected exactly 4 engines");
        ensure!(seed_count > 0, "seed_count must be greater than zero");

        py.allow_threads(move || {
            if let Some(dir) = &self.log_dir {
                fs::create_dir_all(dir)?;
            }

            let n = usize::try_from(seed_count)
                .map_err(|_| anyhow!("seed_count is too large for this platform"))?;
            let seeds = (0..seed_count)
                .map(|offset| {
                    seed_start
                        .0
                        .checked_add(offset)
                        .map(|seed| (seed, seed_start.1))
                        .ok_or_else(|| anyhow!("seed range overflow"))
                })
                .collect::<Result<Vec<_>>>()?;

            let mut agents: Vec<Box<dyn BatchAgent>> = Vec::with_capacity(4);
            for (agent_idx, engine) in engines.into_iter().enumerate() {
                let player_ids: Vec<u8> = (0..n).map(|g| ((agent_idx + g) % 4) as u8).collect();
                agents.push(new_py_agent(engine, &player_ids)?);
            }

            let indexes: Vec<[Index; 4]> = (0..n)
                .map(|g| {
                    array::from_fn(|seat| Index {
                        agent_idx: (seat + 4 - g % 4) % 4,
                        player_id_idx: g,
                    })
                })
                .collect();

            let batch_game = BatchGame::tenhou_hanchan(self.disable_progress_bar);
            let results = batch_game.run(&mut agents, &indexes, &seeds)?;

            if let Some(dir) = &self.log_dir {
                results.iter().try_for_each(|game_result| {
                    let (seed, key) = game_result.seed;
                    let filename: PathBuf = [dir, &format!("{seed}_{key}.json.gz")].iter().collect();

                    let log = game_result.dump_json_log()?;
                    let mut comp = GzEncoder::new(log.as_bytes(), Compression::best());
                    let mut f = File::create(filename)?;
                    io::copy(&mut comp, &mut f)?;
                    Ok(()) as Result<()>
                })?;
            }

            Ok(results
                .into_iter()
                .map(|r| (r.names, r.scores, r.seed))
                .collect())
        })
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn seat_rotation_indexes() {
        // in game g, seat s must map back to agent (s - g) mod 4
        for g in 0..8usize {
            for s in 0..4usize {
                let agent_idx = (s + 4 - g % 4) % 4;
                assert_eq!((agent_idx + g) % 4, s);
            }
        }
    }
}
