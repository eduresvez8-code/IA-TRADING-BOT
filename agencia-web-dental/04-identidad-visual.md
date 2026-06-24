# 04 · Identidad visual — Rutas de logotipo (KALYX)

> Estado: **Ruta 1 elegida (marca tipográfica / wordmark).** Se presentaron 6 rutas de logo (tablero SVG en chat /
> `mockups/`). Eduardo elige una (o una mezcla) y luego se refina el arte final +
> paleta + tipografías + specs de vectorización.
>
> Principio: el logo se diseña **primero en monocromo** (la forma manda); el color
> se afina después. Verde = indicativo del concepto "florecer", no definitivo.
> Insumos de marca: metáfora del **cáliz** (estructura que sostiene la flor),
> **simetría K↔X**, aire **geométrico/corporativo**. Sin emojis. Siempre **KALYX**.

## Las 6 rutas

1. **Marca tipográfica** — el wordmark puro; K y X coloreadas para subrayar la
   simetría. Lo más simple, corporativo y legible. Base segura.
2. **Cáliz que sostiene** ⭐ *(recomendada como símbolo)* — un cáliz/copa que cuida
   una flor (punto) sobre él. La metáfora hecha símbolo; muy ownable.
3. **Monograma K** — una K geométrica que abraza el "brote" en su hueco. Distintivo
   y funciona en tamaño chico (favicon).
4. **Estructura K↔X** — la X como base/cruce que sostiene el brote arriba. El más
   "tech/SpaceX"; sólido y moderno.
5. **Flor geométrica** — un brote saliendo de una base. El más cálido/orgánico;
   habla de crecer.
6. **Ícono de app** — el monograma dentro de un badge oscuro. Demuestra el uso como
   app/avatar (versión en negativo).

## Recomendación de Claude

**Símbolo = Ruta 2 (Cáliz) + wordmark = Ruta 1 + ícono = Ruta 6.** Cuenta la historia
de marca, es poseíble y escala de cartel a favicon. Alternativa más fría/tecnológica:
**Ruta 4 (K↔X)**.

## Dirección de color (preliminar, a confirmar tras elegir logo)

- **Ink / estructura:** `#161A20` (casi negro, base sólida).
- **Verde KALYX / crecimiento:** `#1E8A57`.
- **Brote / acento:** `#2EB16A`.
- Neutros y estados se definen junto con la paleta final en este mismo archivo.

## Pendiente tras la elección

- Refinar el arte de la ruta elegida (proporción, retícula, versiones).
- Paleta de color definitiva (principal, secundaria, neutros, estados) con HEX.
- Tipografías (display + texto) y jerarquía.
- Iconografía e imaginería.
- **Specs de vectorización** + entrega de `mockups/kalyx-logo.svg` limpio.

---

## ✅ DECISIÓN: Ruta 1 — Marca tipográfica (wordmark)

Eduardo eligió la **Ruta 1**. El logo es el nombre **KALYX** en una tipografía
geométrica, con la K y la X como anclas de simetría.

### Nota honesta sobre generar el wordmark con IA (Gemini)
Para un wordmark, los generadores de imagen **no** son la mejor herramienta:
(1) deforman/escriben mal el texto (riesgo alto con un nombre inventado como KALYX);
(2) entregan **raster** (PNG), que igual habría que vectorizar. La vía óptima es
elegir la fuente y entregar **SVG vectorial** directo. El prompt queda como opción.

### Fuentes geométricas candidatas (vía vector recomendada)
- **Space Grotesk** (la del boceto) · **Sora** · **Clash Display** (más afilada,
  "SpaceX") · **Montserrat / Poppins** (más redondeada/amable).

### Prompt para Gemini (image) — guardado por si se usa
```
Logo design: a clean modern wordmark of the brand name "KALYX", spelled exactly with the five uppercase letters K A L Y X. Set in a bold geometric sans-serif typeface with even stroke weight, sharp precise terminals, and balanced, slightly wide letter-spacing. Emphasize the visual symmetry between the first letter "K" and the last letter "X" — both built from crossing diagonal strokes — so the word feels mirrored and stable. Style: minimalist, flat, vector, high-end tech brand identity, corporate and confident. Color: one solid charcoal near-black (#161A20) on a pure flat white background. Horizontal lockup, perfectly centered, generous negative space, crisp clean edges, ultra high resolution. Only the single word KALYX — no tagline, no symbol, no icon, no extra letters, no 3D, no gradient, no shadow, no mockup. Avoid: misspelling, added or missing letters, serif or script fonts, distortion, busy background.
```
Variante bicolor: cambiar la línea de color por *"the letters K and X in emerald
green (#1E8A57), the letters A, L, Y in charcoal near-black (#161A20)"*.

### Siguiente paso
Elegir fuente → Claude entrega `mockups/kalyx-logo.svg` (vector limpio) en sus
versiones (positivo, negativo, monocromo) + paleta final + tipografías + specs.
