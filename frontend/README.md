# Frontend

Next.js 15 application for the automated EDA & ML platform.

## Stack

- **Framework** — Next.js 15 (App Router)
- **Language** — TypeScript
- **Styling** — Tailwind CSS / globals.css

## Development

```bash
npm install
npm run dev
```

App runs at `http://localhost:3000`.

## Structure

```
app/
  page.tsx                     # Dashboard
  datasets/[datasetId]/page.tsx
  runs/[runId]/page.tsx
  runs/page.tsx
components/
  dashboard/                   # Upload + overview widgets
  datasets/                    # EDA section, ML run launcher
  layout/                      # App shell, sidebar
  runs/                        # Run status display
lib/
  api.ts                       # API client
```

## Environment

| Variable | Description |
|---|---|
| `NEXT_PUBLIC_API_URL` | Backend API base URL (default: `http://localhost:8000`) |

## Build

```bash
npm run build
npm start
```

## Lint

```bash
npm run lint
```
