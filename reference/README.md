# reference/

Digitised benchmark data, and the scripts that produce it.

## The source PDFs are deliberately NOT committed

Both papers are third-party publications; this repo stores only the extracted
numeric data and the reproducible extraction scripts. Fetch the PDFs locally
before running the digitisers:

```bash
# Chan & Mittal, CTR Proc. Summer Program 1996, 347-358  (NASA NTRS, public)
curl -L -o reference/chan_mittal_CTR_summer_program_1996.pdf \
     https://ntrs.nasa.gov/api/citations/19970014673/downloads/19970014673.pdf

# Armaly, Durst, Pereira & Schonung, JFM 127 (1983) 473-496
#   Cambridge University Press, copyrighted -- obtain through your own
#   institutional access and save to:
#   reference/armaly_durst_pereira_schonung_JFM_1983.pdf
```

## Extracted data (committed)

| file | content |
|---|---|
| `armaly_fig4_x1_measured.csv` | Armaly fig. 4, measured reattachment `x1/S` vs Re |
| `armaly_fig13a_x1_predicted.csv` | Armaly fig. 13a, their computed `x1/S` vs Re |
| `gartling_re800_x{7,15}_{u,v,omega}.csv` | Gartling benchmark profiles, from Chan fig. 3 |
| `gartling_re800_x{7,15}_profiles.csv` | the same on a common uniform `y` grid |

Regenerate with `armaly_digitize.py` and `gartling_digitize.py`. Both render the
figure pages with ghostscript at 600 dpi and extract by pixel analysis; nothing
is read by eye. See `GARTLING_VALIDATION.md` §3 for the marker-pitch bias that
the Gartling extraction has to correct, and the physics gates used to validate
both.

## Gartling's own tables

Gartling (1990), IJNMF **11** 953-967, is paywalled, and Notus, the SU2
laminar-step tutorial and the Abaqus verification manual all cite it while
publishing only their own results. If you have institutional access, the
tabulated values are preferable to this extraction for any quantitative gate.
