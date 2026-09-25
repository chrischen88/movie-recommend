import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { api, type Health } from "./api";
import MatchReviewPage from "./pages/MatchReviewPage";
import MetricsPage from "./pages/MetricsPage";
import RecommendationsPage from "./pages/RecommendationsPage";
import UploadPage from "./pages/UploadPage";

const navClass = ({ isActive }: { isActive: boolean }) =>
  `shrink-0 whitespace-nowrap px-1.5 sm:px-3 py-1.5 rounded-md text-sm ${isActive ? "bg-zinc-800 text-white" : "text-zinc-400 hover:text-white"}`;

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  return (
    <div className="min-h-screen">
      <header className="border-b border-zinc-800">
        {/* On phones the nav wraps onto its own row and the key badges are hidden. */}
        <div className="mx-auto max-w-6xl px-4 py-3 flex flex-wrap items-center gap-x-4 gap-y-2">
          <span className="font-semibold tracking-tight whitespace-nowrap">Film Picks</span>
          <nav className="order-last sm:order-none w-full sm:w-auto flex gap-0.5 sm:gap-1 overflow-x-auto">
            <NavLink to="/" className={navClass} end>
              Upload
            </NavLink>
            <NavLink to="/recommendations" className={navClass}>
              Recommendations
            </NavLink>
            <NavLink to="/matches" className={navClass}>
              Matches
            </NavLink>
            <NavLink to="/metrics" className={navClass}>
              Metrics
            </NavLink>
          </nav>
          <div className="ml-auto flex gap-2 text-xs">
            {health ? (
              Object.entries(health.features).map(([name, on]) => (
                <span
                  key={name}
                  className={`hidden sm:inline rounded px-2 py-0.5 ${on ? "bg-emerald-900/60 text-emerald-300" : "bg-zinc-800 text-zinc-500"}`}
                  title={on ? `${name} key configured` : `${name} key missing — feature disabled`}
                >
                  {name}
                </span>
              ))
            ) : (
              <span className="text-red-400">API offline</span>
            )}
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-8">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/matches" element={<MatchReviewPage />} />
          <Route path="/recommendations" element={<RecommendationsPage />} />
          <Route path="/metrics" element={<MetricsPage />} />
        </Routes>
      </main>
      <footer className="border-t border-zinc-800">
        <div className="mx-auto max-w-6xl px-4 py-4 text-xs text-zinc-500 space-y-1">
          <p>
            Film data and posters from{" "}
            <a href="https://www.themoviedb.org/" className="underline hover:text-zinc-300">
              TMDB
            </a>
            . This product uses the TMDB API but is not endorsed or certified by TMDB. IMDb, Rotten
            Tomatoes and Metacritic ratings via{" "}
            <a href="https://www.omdbapi.com/" className="underline hover:text-zinc-300">
              OMDb
            </a>
            ; collaborative scores trained on{" "}
            <a href="https://grouplens.org/datasets/movielens/" className="underline hover:text-zinc-300">
              MovieLens
            </a>
            . Not affiliated with Letterboxd.
          </p>
          <p>Your export is processed in memory and never saved on the server.</p>
        </div>
      </footer>
    </div>
  );
}
