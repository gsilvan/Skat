# Local card sprite

The UI works without a sprite and renders a simple CSS fallback deck. To use
the Skatstube French-card sprite locally, run:

```bash
./web/scripts/fetch-cards.sh
```

The downloaded PNG is ignored by Git because its redistribution rights are
unclear. Do not commit it. `Card.tsx` is the single mapping seam for replacing
the image with an original or clearly licensed 8×4 sprite later.
