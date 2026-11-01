# Ride collection endpoint

Why this exists: the model plateaus at ~4.5 m/s speed error because IO-VNBD has
about **five vehicles**, and suspension response is part of the signal it must
learn. Hours of data from one car will not fix that — vehicles from many people
will. This is the cheapest path to that.

## Setup (about 10 minutes, free tier)

```bash
npm install -g wrangler
wrangler login
wrangler r2 bucket create idr-rides
cd cloud && wrangler deploy
wrangler secret put UPLOAD_TOKEN      # invent a long random string
```

Then in the app's SENSORS tab, set the endpoint to:

```
https://idr-collect.<your-subdomain>.workers.dev/<your-token>
```

## Pulling the rides down

```bash
wrangler r2 object get idr-rides/<device>/<file>.csv --file ride.csv
```

## Notes

- Files are grouped by device, so multiple collectors never collide.
- Re-uploading the same name is refused rather than overwriting.
- The app marks a local `.uploaded` sidecar and **never deletes the CSV**, so a
  broken endpoint cannot cost you a drive.
