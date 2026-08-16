use super::{ActionCandidate, PlayerState};
use crate::tile::Tile;

use anyhow::Result;
use pyo3::prelude::*;

#[pymethods]
impl PlayerState {
    #[getter]
    #[inline]
    #[must_use]
    pub const fn player_id(&self) -> u8 {
        self.player_id
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn kyoku(&self) -> u8 {
        self.kyoku
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn honba(&self) -> u8 {
        self.honba
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn kyotaku(&self) -> u8 {
        self.kyotaku
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn is_oya(&self) -> bool {
        self.oya == 0
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn tehai(&self) -> [u8; 34] {
        self.tehai
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn akas_in_hand(&self) -> [bool; 3] {
        self.akas_in_hand
    }

    #[getter]
    #[inline]
    #[must_use]
    pub fn chis(&self) -> &[u8] {
        &self.chis
    }
    #[getter]
    #[inline]
    #[must_use]
    pub fn pons(&self) -> &[u8] {
        &self.pons
    }
    #[getter]
    #[inline]
    #[must_use]
    pub fn minkans(&self) -> &[u8] {
        &self.minkans
    }
    #[getter]
    #[inline]
    #[must_use]
    pub fn ankans(&self) -> &[u8] {
        &self.ankans
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn at_turn(&self) -> u8 {
        self.at_turn
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn shanten(&self) -> i8 {
        self.shanten
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn waits(&self) -> [bool; 34] {
        self.waits
    }

    #[inline]
    #[pyo3(name = "last_self_tsumo")]
    fn last_self_tsumo_py(&self) -> Option<String> {
        self.last_self_tsumo.map(|t| t.to_string())
    }
    #[inline]
    #[pyo3(name = "last_kawa_tile")]
    fn last_kawa_tile_py(&self) -> Option<String> {
        self.last_kawa_tile.map(|t| t.to_string())
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn last_cans(&self) -> ActionCandidate {
        self.last_cans
    }

    #[inline]
    #[pyo3(name = "ankan_candidates")]
    fn ankan_candidates_py(&self) -> Vec<String> {
        self.ankan_candidates
            .iter()
            .map(|t| t.to_string())
            .collect()
    }
    #[inline]
    #[pyo3(name = "kakan_candidates")]
    fn kakan_candidates_py(&self) -> Vec<String> {
        self.kakan_candidates
            .iter()
            .map(|t| t.to_string())
            .collect()
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn can_w_riichi(&self) -> bool {
        self.can_w_riichi
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn self_riichi_declared(&self) -> bool {
        self.riichi_declared[0]
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn self_riichi_accepted(&self) -> bool {
        self.riichi_accepted[0]
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn at_furiten(&self) -> bool {
        self.at_furiten
    }

    #[pyo3(name = "real_time_shanten")]
    #[must_use]
    fn real_time_shanten_py(&self) -> i8 {
        self.real_time_shanten()
    }

    #[pyo3(name = "reaction_summary")]
    fn reaction_summary_py(&self, mjai_json: &str) -> Result<(i8, [bool; 34], bool)> {
        self.validate_reaction_json(mjai_json)?;
        let mut after = self.clone();
        after.update_json(mjai_json)?;
        Ok((after.real_time_shanten(), after.waits, after.at_furiten))
    }

    #[pyo3(name = "reaction_analysis")]
    fn reaction_analysis_py(
        &self,
        mjai_json: &str,
    ) -> Result<(i8, [bool; 34], bool, u32, bool, bool)> {
        self.validate_reaction_json(mjai_json)?;
        let mut after = self.clone();
        after.update_json(mjai_json)?;
        Ok((
            after.real_time_shanten(),
            after.waits,
            after.at_furiten,
            after.ukeire(),
            after.wait_has_yaku(),
            after.is_menzen,
        ))
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn tiles_seen(&self) -> [u8; 34] {
        self.tiles_seen
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn discarded_tiles(&self) -> [bool; 34] {
        self.discarded_tiles
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn tiles_left(&self) -> u8 {
        self.tiles_left
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn is_all_last(&self) -> bool {
        self.is_all_last
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn is_menzen(&self) -> bool {
        self.is_menzen
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn scores(&self) -> [i32; 4] {
        self.scores
    }

    #[inline]
    #[pyo3(name = "bakaze")]
    fn bakaze_py(&self) -> String {
        self.bakaze.to_string()
    }

    #[inline]
    #[pyo3(name = "jikaze")]
    fn jikaze_py(&self) -> String {
        self.jikaze.to_string()
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn riichi_declared(&self) -> [bool; 4] {
        self.riichi_declared
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn riichi_accepted(&self) -> [bool; 4] {
        self.riichi_accepted
    }

    #[pyo3(name = "ukeire")]
    #[must_use]
    fn ukeire_py(&self) -> u32 {
        self.ukeire()
    }

    #[pyo3(name = "wait_has_yaku")]
    #[must_use]
    fn wait_has_yaku_py(&self) -> bool {
        self.wait_has_yaku()
    }
}

impl PlayerState {
    #[inline]
    #[must_use]
    pub const fn last_self_tsumo(&self) -> Option<Tile> {
        self.last_self_tsumo
    }
    #[inline]
    #[must_use]
    pub const fn last_kawa_tile(&self) -> Option<Tile> {
        self.last_kawa_tile
    }

    #[inline]
    #[must_use]
    pub fn ankan_candidates(&self) -> &[Tile] {
        &self.ankan_candidates
    }
    #[inline]
    #[must_use]
    pub fn kakan_candidates(&self) -> &[Tile] {
        &self.kakan_candidates
    }
}
