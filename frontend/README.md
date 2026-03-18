# Used Car Price Prediction Frontend

This frontend is a React + Vite app for a used car price prediction system. It collects vehicle details from the user, sends them to the FastAPI backend, and displays the predicted resale price.

## Tech Stack

- React 19
- Vite 8
- Axios

## Prerequisites

- Node.js 18+
- npm
- Backend API running locally or reachable through `VITE_API_BASE_URL`

## Install

```bash
npm install
```

## Run In Development

```bash
npm run dev
```

By default, the frontend expects the backend at:

```text
http://127.0.0.1:8000
```

To point the frontend to a different backend, create a `.env.local` file inside `frontend/`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Build

```bash
npm run build
```

## Lint

```bash
npm run lint
```

## App Flow

1. User enters car details in the form.
2. Frontend sends a `POST /predict` request to the backend.
3. Backend returns the predicted price.
4. Frontend shows the estimated result or an error message.

## Main Files

- `src/App.jsx`: page layout and hero section
- `src/components/CarForm.jsx`: form state, validation, and API call
- `src/App.css`: component styling
- `src/index.css`: global styles
