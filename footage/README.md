Put the ORIGINAL camera files here (any sub-folders are fine): .MOV / .MP4 straight from the phone or camera.
They are only ever read by the tools — never modified or re-encoded in place.

- Clips may share a file name, in different sub-folders (two phones, a counter that wrapped) or with different
  extensions (IMG_0001.MOV and IMG_0001.mp4). `tools/review.py` names each clip's sheets and proxy
  `<stem>_<6-character hash of its path>`, e.g. footage/IMG_0001.MOV → `work/review/IMG_0001_5d07bf/` and
  `work/proxies/IMG_0001_5d07bf.mp4`. Every sheet's title shows the clip's full path. After moving or
  renaming a clip, run the review again.
- Variable-frame-rate phone clips are fine. Review times (contact-sheet labels, metrics, suggested moments)
  are the clip's real frame timestamps, in the same seconds as the EDL's `src_in`.
- Dolby Vision **profile 5** files (no HLG- or SDR-compatible base layer) cannot be converted. The review
  marks their previews "[DV5: COLOURS WRONG]" and render.py refuses them, so export them as HLG or SDR first.
  iPhone HDR video (Dolby Vision 8.4, HLG base layer) is fine as it is.
