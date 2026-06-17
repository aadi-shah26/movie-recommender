import { cn } from "@/lib/utils";

interface MoviePosterProps {
  title: string;
  year?: number | null;
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
  const words = title.replace(/[^A-Za-z0-9 ]/g, "").trim().split(/\s+/);
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[1][0]).toUpperCase();
}

const SIZE_CLASSES = {
  sm: "h-14 w-10 text-sm rounded-md",
  md: "h-20 w-14 text-base rounded-lg",
  lg: "h-28 w-20 text-xl rounded-xl",
} as const;

export function MoviePoster({ title, year, size = "sm", className }: MoviePosterProps) {
  const h = hash(title);
  const [from, to] = PALETTES[h % PALETTES.length];
  const angle = (h % 6) * 30 + 120;

  return (
    <div
      className={cn(
        "relative shrink-0 overflow-hidden flex items-center justify-center font-bold text-white shadow-md",
        SIZE_CLASSES[size],
        className,
      )}
      style={{ backgroundImage: `linear-gradient(${angle}deg, ${from}, ${to})` }}
      aria-hidden
    >
      <div className="absolute inset-0 opacity-30 mix-blend-overlay bg-[radial-gradient(circle_at_30%_20%,white,transparent_60%)]" />
      <span className="relative drop-shadow">{initials(title)}</span>
      {year && size !== "sm" && (
        <span className="absolute bottom-1 right-1.5 text-[9px] font-medium opacity-80">
          {year}
        </span>
      )}
    </div>
  );
}
