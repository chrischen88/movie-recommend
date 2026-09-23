import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { api, type Health } from "./api";
import MatchReviewPage from "./pages/MatchReviewPage";
import RecommendationsPage from "./pages/RecommendationsPage";
import UploadPage from "./pages/UploadPage";

const navClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-1.5 rounded-md text-sm ${isActive ? "bg-zinc-800 text-white" : "text-zinc-400 hover:text-white"}`;

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  return (
    <div className="min-h-screen">
      <header className="border-b border-zinc-800">
        <div className="mx-auto max-w-6xl px-4 py-3 flex items-center gap-4">
          <span className="font-semibold tracking-tight">Film Picks</span>
          <nav className="flex gap-1">
            <NavLink to="/recommendations" className={navClass}>
              Recommendations
            </NavLink>
            <NavLink to="/" className={navClass} end>
              Upload
            </NavLink>
            <NavLink to="/matches" className={navClass}>
              Matches
            </NavLink>
          </nav>
          <div className="ml-auto flex gap-2 text-xs">
            {health ? (
              Object.entries(health.features).map(([name, on]) => (
                <span
                  key={name}
                  className={`rounded px-2 py-0.5 ${on ? "bg-emerald-900/60 text-emerald-300" : "bg-zinc-800 text-zinc-500"}`}
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
        </Routes>
      </main>
    </div>
  );
}
