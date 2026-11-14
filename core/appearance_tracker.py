"""Reuse our OSNet features for short-term association, not a second encoder."""
import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.utils import matching


class AppearanceTracker(BOTSORT):
    # A few missing detections should not force an already changing crop back
    # to its old EMA. Beyond this brief gap require the stricter lost-track gate.
    SHORT_GAP_FRAMES = 5

    DRIFT_LAG = 15

    def __init__(self, args, minimum_similarity=0.5, drift_threshold=0.0):
        super().__init__(args)
        self.minimum_similarity = minimum_similarity
        self.drift_threshold = drift_threshold
        self.identity_separations = {}
        self._drift_rejections = {}

    def set_detection_threshold(self, confidence):
        """Weak boxes may continue a track, but cannot create a new person."""
        self.args.track_high_thresh = max(self.args.track_high_thresh, confidence)
        self.args.new_track_thresh = max(self.args.new_track_thresh, confidence)

    @property
    def detection_floor(self):
        return min(self.args.track_high_thresh, self.args.track_low_thresh)

    def update(self, results, img=None, feats=None, **kwargs):
        self._image_shape = results.orig_shape
        self._drift_rejections = {}
        output = super().update(results, img=img, feats=feats, **kwargs)
        height = results.orig_shape[0]
        for track in list(self.tracked_stracks):
            if track.frame_id != self.frame_id:
                continue
            if (self.drift_threshold > 0 and track.curr_feat is not None
                    and track.score >= self.args.track_high_thresh):
                history = getattr(track, '_appearance_history', [])
                history.append((self.frame_id, track.curr_feat.copy()))
                track._appearance_history = [sample for sample in history
                                             if self.frame_id - sample[0] <= self.DRIFT_LAG + 5]
                track._last_strong_box = np.asarray(results.xyxy[int(track.idx)]).copy()
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
        self.identity_separations = {
            t.track_id: t._separated_from for t in self.tracked_stracks
            if t.frame_id == self.frame_id and getattr(t, '_separated_from', ())}
        return output

    def reset(self):
        super().reset()
        self.identity_separations = {}
        self._drift_rejections = {}

    def get_dists(self, tracks, detections):
        geometry = matching.iou_distance(tracks, detections)
        distances = matching.fuse_score(geometry, detections) if self.args.fuse_score else geometry.copy()
        distances = self._gate_appearance(tracks, detections, distances, blend=True)
        # A similar crop must not teleport a trajectory across the image.
        distances[geometry > 1 - self.proximity_thresh] = 1.0
        active = [i for i, t in enumerate(tracks)
                  if t.state == TrackState.Tracked and self.frame_id - t.frame_id <= 1]
        older_lost = [i for i, t in enumerate(tracks)
                      if t.state == TrackState.Lost and 3 < self.frame_id - t.frame_id <= self.SHORT_GAP_FRAMES]
        if active and older_lost:
            # A cropped outgoing person may contain the next passenger in its
            # last box. Do not reactivate that lost ID over an uninterrupted,
            # appearance-compatible trajectory already following this body.
            protected = np.flatnonzero(np.any((geometry[active] < .5)
                                             & (distances[active] < self.args.match_thresh), axis=0))
            distances[np.ix_(older_lost, protected)] = 1.0
        return distances

    def _gate_drift(self, tracks, detections, distances, *, blend):
        if self.drift_threshold <= 0:
            return distances
        height, width = getattr(self, '_image_shape', (1, 1))
        # A weak, almost frame-sized box may have merged neighbouring bodies.
        # Comparing only adjacent crops then lets appearance drift gradually
        # onto the next person. Do not constrain normal-sized or confident
        # detections: their appearance legitimately changes during a turn.
        valid = [i for i, d in enumerate(detections)
                 if d.curr_feat is not None and d.score < .5
                 and d.xyxy[2] - d.xyxy[0] >= .90 * width
                 and d.xyxy[3] - d.xyxy[1] >= .75 * height]
        if not valid:
            return distances
        current = np.stack([detections[i].curr_feat for i in valid])
        limit = self.args.match_thresh if blend else .5
        peers = [t for t in [*self.tracked_stracks, *self.lost_stracks]
                 if len(getattr(t, '_appearance_history', [])) >= 5
                 and self.frame_id - t._appearance_history[-1][0] <= 5
                 and hasattr(t, '_last_strong_box') and t.curr_feat is not None]
        for i, track in enumerate(tracks):
            if self.frame_id - track.frame_id > self.SHORT_GAP_FRAMES:
                continue
            history = [vector for frame, vector in getattr(track, '_appearance_history', [])
                       if abs(self.frame_id - frame - self.DRIFT_LAG) <= 2]
            if len(history) < 3:
                continue
            similarity = np.median(np.stack(history) @ current.T, axis=0)
            for k in np.flatnonzero(similarity < self.drift_threshold):
                j = valid[k]
                if (blend and matching.iou_distance([track], [detections[j]])[0, 0]
                        > 1 - self.proximity_thresh):
                    continue
                if distances[i, j] < limit and self._has_distinct_peer(track, detections[j], peers):
                    self._drift_rejections.setdefault(int(detections[j].idx), set()).add(track.track_id)
                    distances[i, j] = 1.0
        return distances

    @staticmethod
    def _has_distinct_peer(track, detection, peers):
        # A single person bending close to the camera also changes the whole
        # crop. Require recent evidence of a different overlapping person;
        # a new/noisy one-frame detection is not sufficient evidence to split.
        box = detection.xyxy
        area = float(np.prod(box[2:] - box[:2]))
        for peer in peers:
            if peer.track_id == track.track_id or float(peer.curr_feat @ detection.curr_feat) > .80:
                continue
            other = peer._last_strong_box
            intersection = float(np.prod(np.maximum(0, np.minimum(box[2:], other[2:])
                                                    - np.maximum(box[:2], other[:2]))))
            union = area + float(np.prod(other[2:] - other[:2])) - intersection
            if intersection / max(1, union) >= .05:
                return True
        return False

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
            previous = np.stack([tracks[i].curr_feat if ages[j] <= self.SHORT_GAP_FRAMES and tracks[i].curr_feat is not None
                                 else tracks[i].smooth_feat for j, i in enumerate(valid_tracks)])
            current = np.stack([detections[i].curr_feat for i in valid_detections])
            similarity = np.clip(previous @ current.T, -1.0, 1.0)
            if blend:
                # Geometry and appearance must agree. Taking their minimum
                # lets one attractive partial/duplicate crop defeat a better
                # full-body detection, regardless of its confidence and motion.
                grid = np.ix_(valid_tracks, valid_detections)
                distances[grid] = .7 * distances[grid] + .3 * (1 - similarity)
            minimum = np.where(ages <= self.SHORT_GAP_FRAMES, self.minimum_similarity,
                               max(.85, self.minimum_similarity))
            rows, cols = np.where(similarity < minimum[:, None])
            distances[np.asarray(valid_tracks)[rows], np.asarray(valid_detections)[cols]] = 1.0
            # Prefer a currently observed trajectory over an old lost track
            # when both predict the same location. Never reward missing time.
            distances[np.ix_(valid_tracks, valid_detections)] += np.minimum(.2, np.maximum(0, ages - 1) * .01)[:, None]
        return self._gate_drift(tracks, detections, distances, blend=blend)

    def _second_association(self, pool, unmatched, detections, activated, refind, lost):
        # ByteTrack's low-confidence stage otherwise bypasses get_dists and
        # can overwrite a person's ID using IoU alone. Keep its unfused IoU
        # cost/threshold, but apply the same appearance veto as the first stage.
        remaining = [pool[i] for i in unmatched if pool[i].state == TrackState.Tracked
                     or (pool[i].state == TrackState.Lost and self.frame_id - pool[i].frame_id <= 3)]
        distances = self._gate_appearance(remaining, detections, matching.iou_distance(remaining, detections))
        matches, unmatched, _ = matching.linear_assignment(distances, thresh=.5)
        self._apply_matches(matches, remaining, detections, activated, refind)
        for i in unmatched:
            if remaining[i].state == TrackState.Tracked:
                remaining[i].mark_lost()
                lost.append(remaining[i])

    def _init_new_tracks(self, unmatched, detections, activated, refind=None):
        before = len(activated)
        super()._init_new_tracks(unmatched, detections, activated, refind)
        new_tracks = activated[before:]
        for track in new_tracks:
            track._separated_from = tuple(sorted(self._drift_rejections.get(int(track.idx), ())))
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
