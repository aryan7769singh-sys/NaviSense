/**
 * Cloudflare Worker: receives ride recordings from the IDR Logger and stores them
 * in R2. Deploy with `npx wrangler deploy`.
 *
 * The phone does `PUT https://<worker>/<filename>` with the CSV as the body.
 *
 * wrangler.toml needs:
 *
 *   name = "idr-collect"
 *   main = "worker.js"
 *   compatibility_date = "2026-01-01"
 *   [[r2_buckets]]
 *   binding = "RIDES"
 *   bucket_name = "idr-rides"
 *
 * Then set the shared secret:  npx wrangler secret put UPLOAD_TOKEN
 * and point the app at:        https://idr-collect.<subdomain>.workers.dev/<token>
 */
export default {
  async fetch(request, env) {
    if (request.method !== "PUT") {
      return new Response("PUT a .csv to /<token>/<filename>", { status: 405 });
    }

    const url = new URL(request.url);
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts.length !== 2) return new Response("Bad path", { status: 400 });

    const [token, name] = parts;
    // Without this, anyone who finds the URL can fill the bucket.
    if (token !== env.UPLOAD_TOKEN) return new Response("Forbidden", { status: 403 });
    if (!/^[\w.\-]+\.csv$/.test(name)) return new Response("Bad filename", { status: 400 });

    const device = request.headers.get("X-Device") || "unknown";
    // Group by device so one phone's rides never overwrite another's, and so
    // vehicle diversity is visible in the bucket layout itself.
    const key = `${device}/${name}`;

    // Never silently overwrite: a repeated upload of a different drive with the
    // same timestamp would destroy data that cannot be recollected.
    const existing = await env.RIDES.head(key);
    if (existing) return new Response("Already stored", { status: 200 });

    await env.RIDES.put(key, request.body, {
      httpMetadata: { contentType: "text/csv" },
      customMetadata: { device, uploadedAt: new Date().toISOString() },
    });

    return new Response(JSON.stringify({ ok: true, key }), {
      headers: { "content-type": "application/json" },
    });
  },
};
