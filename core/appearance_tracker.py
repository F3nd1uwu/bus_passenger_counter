"""Reuse our OSNet features for short-term association, not a second encoder."""
import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.utils import matching


class AppearanceTracker(BOTSORT):
    def __init__(self, args, minimum_similarity=0.5):
        super().__init__(args)
        self.minimum_similarity = minimum_similarity

    def update(self, results, img=None, feats=None, **kwargs):
        output = super().update(results, img=img, feats=feats, **kwargs)
        height = results.orig_shape[0]
        for track in list(self.tracked_stracks):
            if track.frame_id != self.frame_id:
                continue
            box = results.xyxy[int(track.idx)]
            observed_height = float(box[3] - box[1])
            largest = max(getattr(track, '_largest_visible_height', 0), observed_height)
            track._largest_visible_height = largest
            # A formerly full-sized body shrinking to a small clipped fragment
            # while moving out of the image has finished its visible trajectory.
            # Leaving it available for matching lets the next boarding person
            # inherit its ID as soon as the doorway fills again. The counter's
            # hourly identity remains available if the original person returns.
            moving_out = ((box[1] <= height * .02 and track.mean[5] < -1)
                          or (box[3] >= height * .98 and track.mean[5] > 1))
            if largest >= height * .45 and observed_height < largest * .30 and moving_out:
                track.mark_removed()
                self.tracked_stracks.remove(track)
                self.removed_stracks.append(track)
        return output

    def get_dists(self, tracks, detections):
        geometry = matching.iou_distance(tracks, detections)
        distances = matching.fuse_score(geometry, detections) if self.args.fuse_score else geometry.copy()
        distances = self._gate_appearance(tracks, detections, distances, blend=True)
        # A similar crop must not teleport a trajectory across the image.
        distances[geometry > 1 - self.proximity_thresh] = 1.0
        return distances

    def _gate_appearance(self, tracks, detections, distances, *, blend=False):
        if not tracks or not detections:
            return distances
        # Use the last view for an uninterrupted trajectory. A moving person
        # can grow from a small, door-occluded crop to a full body in a few
        # frames; the slow EMA still describes the initial partial crop.
        # Lost trajectories need stronger evidence before being reactivated:
        # the next passenger often occupies exactly the same doorway box.
        valid_tracks = [i for i, t in enumerate(tracks) if t.smooth_feat is not None]
        valid_detections = [i for i, d in enumerate(detections) if d.curr_feat is not None]
        if valid_tracks and valid_detections:
            ages = np.asarray([max(0, self.frame_id - tracks[i].frame_id) for i in valid_tracks])
            previous = np.stack([tracks[i].curr_feat if ages[j] <= 3 and tracks[i].curr_feat is not None
                                 else tracks[i].smooth_feat for j, i in enumerate(valid_tracks)])
            current = np.stack([detections[i].curr_feat for i in valid_detections])
            similarity = np.clip(previous @ current.T, -1.0, 1.0)
            if blend:
                # Geometry and appearance must agree. Taking their minimum
                # lets one attractive partial/duplicate crop defeat a better
                # full-body detection, regardless of its confidence and motion.
                grid = np.ix_(valid_tracks, valid_detections)
                distances[grid] = .7 * distances[grid] + .3 * (1 - similarity)
            minimum = np.where(ages <= 3, self.minimum_similarity, max(.85, self.minimum_similarity))
            rows, cols = np.where(similarity < minimum[:, None])
            distances[np.asarray(valid_tracks)[rows], np.asarray(valid_detections)[cols]] = 1.0
            # Prefer a currently observed trajectory over an old lost track
            # when both predict the same location. Never reward missing time.
            distances[np.ix_(valid_tracks, valid_detections)] += np.minimum(.2, np.maximum(0, ages - 1) * .01)[:, None]
        return distances

    def _second_association(self, pool, unmatched, detections, activated, refind, lost):
        # ByteTrack's low-confidence stage otherwise bypasses get_dists and
        # can overwrite a person's ID using IoU alone. Keep its unfused IoU
        # cost/threshold, but apply the same appearance veto as the first stage.
        remaining = [pool[i] for i in unmatched if pool[i].state == TrackState.Tracked]
        distances = self._gate_appearance(remaining, detections, matching.iou_distance(remaining, detections))
        matches, unmatched, _ = matching.linear_assignment(distances, thresh=.5)
        self._apply_matches(matches, remaining, detections, activated, refind)
        for i in unmatched:
            remaining[i].mark_lost()
            lost.append(remaining[i])

    def _init_new_tracks(self, unmatched, detections, activated, refind=None):
        before = len(activated)
        super()._init_new_tracks(unmatched, detections, activated, refind)
        new_tracks = activated[before:]
        old_tracks = {t.track_id: t for t in [*self.lost_stracks, *self.tracked_stracks]
                      if t.state == TrackState.Lost}
        old_tracks = list(old_tracks.values())
        if not new_tracks or not old_tracks:
            return
        overlap = matching.iou_distance(old_tracks, new_tracks)
        gated = self._gate_appearance(old_tracks, new_tracks, overlap.copy())
        # Upstream pool de-duplication uses IoU alone and keeps the older
        # trajectory, even when appearance rejected that association. Without
        # retiring the incompatible lost trajectory it deletes the new person's
        # unconfirmed track on every frame. Hourly identities live in the
        # counter and are not removed here.
        superseded = np.flatnonzero(np.any((overlap < .15) & (gated >= 1), axis=1))
        for i in superseded:
            old_tracks[i].mark_removed()
            self.removed_stracks.append(old_tracks[i])
        self.lost_stracks = [t for t in self.lost_stracks if t.state != TrackState.Removed]
