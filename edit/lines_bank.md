# Arabic text — candidates (original writing)

Final choice is made **against the actual images** (2–4 text moments in total, title included).
All lines are original; no quotations, no historical claims, no location labels.
Religious wording is kept out of the invented lines on purpose — the place speaks for itself.

## Title
- **المدينة المنوّرة** — fixed (as written by the filmmaker, with the shadda).

## Arrival / opening
- هنا… يهدأ كلُّ شيء — *Here… everything grows still.*
- وصلتُ… فاطمأنّ القلب — *I arrived… and the heart found rest.*
- طيبة… حيثُ يطمئنّ القلب — *Taybah… where the heart is at peace.* (Taybah is a well-known name of Madinah)

## Beauty / scale (wide architecture, light, sky)
- نورٌ يملأ المكان — *A light that fills the place.*
- سكينةٌ تسكنُ القلب — *A serenity that settles in the heart.*
- في كلِّ زاويةٍ… سكينة — *In every corner… serenity.*

## Intimate details (marble, lamps, hands, footsteps, umbrellas)
- خطواتٌ هادئة… وقلبٌ ممتلئ — *Quiet steps… and a full heart.*
- تفاصيلُ صغيرة… تبقى طويلاً — *Small details… that stay for long.*
- نورٌ على الرخام، وطمأنينةٌ في القلب — *Light on the marble, calm in the heart.* (only if marble is on screen)

## Closing / attachment
- أعودُ إليها… وكأنّي لم أغادر — *I return to her… as if I never left.*
- وفي القلبِ حنينٌ لا ينتهي — *And in the heart, a longing that never ends.*
- إلى لقاءٍ قريب — *Until we meet again, soon.*

## Typographic rules used by the renderer
These are the defaults in `tools/render.py` → `build_ass` (the values below are the code's at the time of
writing). A title's own keys in `edit/edl.json`, or `typography.styles`, override them. If this list and
the code ever disagree, the code is what renders.
- Styles (two families only): **title** Amiri Bold 340, #F7F2E8; **line** IBM Plex Sans Arabic Regular 88,
  #F4EFE6; **small** IBM Plex Sans Arabic Medium 66, #EDE6DA.
- libass with HarfBuzz + FriBidi, style Encoding −1 (automatic base direction). Every Arabic line also gets a
  leading U+200F RLM, so a line that starts with a Latin word or digits still reads right-to-left and «.» «،»
  land on the correct side.
- No letter-spacing on Arabic (it breaks joining; `\fsp` is refused). Ellipsis «…» and Arabic comma «،».
- Legibility without a box: two shadow layers in warm near-black #100C0A under the text. The first is a wide,
  soft, low-opacity halo, given as bord / blur: title 16 / 28, line 10 / 18, small 9 / 16, at alpha 0x90
  (≈ 44 % opacity). The second is a tight contact edge: title 3 / 4, line 2 / 2.5, small 2 / 2, at alpha 0x50
  (≈ 69 % opacity).
- Fades: 0.8 s in, 0.7 s out (`fade_in` / `fade_out`; the template's title uses 0.9 / 0.7).
- Drift: `drift` 10 → `\move` from y + 10 to y − 3 over the title's whole time on screen. That is 13 px of
  upward travel (1.3 × drift), ending 3 px above `y`. `drift: 0` holds the title still.
- Soft focus-in: blur 3 → 0.3 (`blur_from` → `blur`) over 1.2 × the fade-in (0.96 s at the default).
- Safe area on 1080×1920: x 90–910, y 250–1480.
- Fonts: only the files in `assets/fonts` are used, and render.py stops if libass would fall back to any other
  font. **Noto Naskh Arabic is a variable font** (`NotoNaskhArabic-VF.ttf`), and libass 0.17 uses only its
  default Regular instance:
  - `bold: 1` on it is synthetic emboldening of the Regular outlines. The font check does not catch this.
  - The named weights ("Noto Naskh Arabic Medium / SemiBold / Bold") cannot be selected; the font check
    stops the render.

  Use Noto Naskh only at Regular weight, or first add the static `NotoNaskhArabic-Bold.ttf` from the
  official static build to `assets/fonts`. The default styles do not use it.
