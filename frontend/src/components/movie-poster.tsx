import { Film } from "lucide-react";
import { cn } from "@/lib/utils";

interface MoviePosterProps {
  title: string;
  year?: number | null;
  /** Optional real poster image URL; falls back to a generated tile. */
  posterUrl?: string | null;
  size?: "sm" | "md" | "lg";
  className?: string;
}

function hash(str: string) {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = (h << 5) - h + str.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h);
}

const PALETTES: [string, string][] = [
  ["#7c3aed", "#ec4899"],
  ["#0ea5e9", "#6366f1"],
  ["#f59e0b", "#ef4444"],
  ["#10b981", "#0ea5e9"],
  ["#f43f5e", "#8b5cf6"],
  ["#06b6d4", "#3b82f6"],
  ["#84cc16", "#14b8a6"],
  ["#a855f7", "#3b82f6"],
  ["#fb7185", "#f59e0b"],
];

function initials(title: string) {
  const words = title
    .replace(/[^A-Za-z0-9 ]/g, "")
    .trim()
    .split(/\s+/);
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[1][0]).toUpperCase();
}

const SIZE_CLASSES = {
  sm: "h-16 w-11 rounded-md",
  md: "h-24 w-16 rounded-lg",
  lg: "h-32 w-[5.5rem] rounded-xl",
} as const;

const INITIALS_CLASSES = {
  sm: "text-sm",
  md: "text-lg",
  lg: "text-2xl",
} as const;

// Faint film-grain so the flat gradients read as textured "art".
const GRAIN =
  "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='80'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.5'/%3E%3C/svg%3E\")";

export function MoviePoster({ title, year, posterUrl, size = "sm", className }: MoviePosterProps) {
  const h = hash(title);
  const [from, to] = PALETTES[h % PALETTES.length];
  const angle = (h % 6) * 30 + 120;

  return (
    <div
      className={cn(
        "group/poster relative shrink-0 overflow-hidden flex items-center justify-center font-bold text-white",
        "ring-1 ring-white/10 shadow-[0_6px_16px_-6px_rgba(0,0,0,0.7)]",
        SIZE_CLASSES[size],
        className,
      )}
      style={
        posterUrl ? undefined : { backgroundImage: `linear-gradient(${angle}deg, ${from}, ${to})` }
      }
      aria-hidden
    >
      {posterUrl ? (
        <img
          src={posterUrl}
          alt=""
          loading="lazy"
          className="absolute inset-0 h-full w-full object-cover"
        />
      ) : (
        <>
          {/* grain + highlight + vignette layers for depth */}
          <div
            className="absolute inset-0 opacity-[0.12] mix-blend-overlay"
            style={{ backgroundImage: GRAIN }}
          />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_18%,rgba(255,255,255,0.45),transparent_55%)]" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/55 via-transparent to-transparent" />
          <Film
            className={cn(
              "absolute right-1 top-1 text-white/30",
              size === "sm" ? "h-3 w-3" : "h-3.5 w-3.5",
            )}
          />
          <span
            className={cn(
              "relative drop-shadow-[0_1px_3px_rgba(0,0,0,0.5)]",
              INITIALS_CLASSES[size],
            )}
          >
            {initials(title)}
          </span>
          {year && size !== "sm" && (
            <span className="absolute bottom-1 right-1.5 text-[9px] font-medium opacity-80">
              {year}
            </span>
          )}
        </>
      )}
    </div>
  );
}
