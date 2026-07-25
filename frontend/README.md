# RocketAttendance — Frontend (React + Vite)

Premium glassmorphism React frontend, served into Django via **django-vite**.
This is being migrated page-by-page from the legacy Django templates. First page: the marketing **home page**.

## Two ways to run

### A) Normal — single server (serves the built bundle)

By default Django serves the **built** React bundle from `frontend/dist`, so you only
need ONE process:

```bash
# first time (from attendence_app_backend/frontend/)
npm install
npm run build            # emits dist/ + dist/.vite/manifest.json

# then (from attendence_app_backend/)
python manage.py runserver
```

Open **http://127.0.0.1:8000/home-v2/**

> After editing anything under `frontend/src`, re-run `npm run build` **and restart
> `runserver`** (django-vite caches the manifest at startup) to see the change.

### B) Live development — hot reload (two servers)

For fast iteration with instant HMR, run Vite too and start Django with the dev flag:

```bash
# terminal 1 (from attendence_app_backend/frontend/)
npm run dev

# terminal 2 (from attendence_app_backend/) — Git Bash
DJANGO_VITE_DEV=1 python manage.py runserver
#   PowerShell:  $env:DJANGO_VITE_DEV=1; python manage.py runserver
```

Now `frontend/src` edits hot-reload in the browser with no rebuild/restart.

## Production

```bash
npm run build
python manage.py collectstatic
```
`collectstatic` copies the hashed `dist/` assets into `STATIC_ROOT` for your web server.

## Structure

```
src/
  main.tsx            entry (mounts <App/> into #root)
  index.css           Tailwind v4 + glassmorphism design tokens
  App.tsx             renders <HomePage/>
  lib/                helpers (cn, hooks, mock data)
  components/ui/      glass primitives (GlassCard, GlassButton, Marquee, Timeline, ...)
  sections/           home page sections
  home/HomePage.tsx   composes all sections
```
