# Vercel Deployment

CancerHawk frontend is a Next.js static site hosted on Vercel.

## One-time Setup
1. Go to [vercel.com/import](https://vercel.com/import)
2. Import `asimog/cancerhawk` GitHub repo
3. Framework: **Next.js**
4. Build command: `npm run build`
5. Output directory: `.next`
6. Add environment variable:
   ```
   NEXT_PUBLIC_BACKEND_URL=https://cancerhawk-production.up.railway.app
   ```
7. Deploy

## Configuration
```json
// vercel.json
{
  "framework": "nextjs",
  "buildCommand": "npm run build",
  "outputDirectory": ".next"
}
```

## Auto-deploy
Vercel auto-deploys on every push to `master`. No manual trigger needed.

## Routes
| Path | Purpose |
|---|---|
| `/` | Home page with route cards |
| `/current-block` | Latest block with simulations |
| `/previous-blocks` | Block archive |
| `/run-research` | Research run UI |
| `/autonomous-logs` | Live event stream |
| `/jobs` | Job feed |
| `/jobs/:id` | Job detail |
| `/music` | Audio-reactive orb |
| `/llms.txt` | AI-discoverable docs |

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `NEXT_PUBLIC_BACKEND_URL` | yes | Railway worker URL |

## Post-deploy Verification
```bash
curl -s https://cancerhawk.org | grep CancerHawk
curl -s https://cancerhawk.org/api/health
# → (proxied to Railway worker)
```

## Build
```bash
npm install
npm run build      # next build --webpack
npm run lint       # tsc --noEmit
```

## Note
The Vercel site does NOT have access to `OPENROUTER_API_KEY` or any server secrets. API calls are proxied to the Railway worker which holds credentials.
