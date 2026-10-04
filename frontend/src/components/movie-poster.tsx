import { cn } from "@/lib/utils";

interface MoviePosterProps {
  title: string;
  year?: number | null;
  /** Optional real poster image URL; falls back to a typeset title card. */
  posterUrl?: string | null;
  size?: "sm" | "md" | "lg" | "fill";
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

// Muted film-stock tones for posterless titles: they should sit quietly next
// to real poster art, not compete with it.
const TONES = [
  "oklch(0.3 0.025 60)",
  "oklch(0.28 0.02 150)",
  "oklch(0.29 0.025 25)",
  "oklch(0.27 0.02 240)",
  "oklch(0.31 0.015 90)",
];

const SIZE_CLASSES = {
  sm: "h-[3.75rem] w-10 rounded-[3px]",
  md: "h-24 w-16 rounded-sm",
  lg: "h-36 w-24 rounded",
  fill: "aspect-[2/3] w-full rounded",
} as const;

const TITLE_CLASSES = {
  sm: "text-[8px] leading-[1.05] p-1",
  md: "text-xs leading-tight p-1.5",
  lg: "text-sm leading-tight p-2",
  fill: "text-xl leading-tight p-4",
} as const;

export function MoviePoster({ title, year, posterUrl, size = "sm", className }: MoviePosterProps) {
  return (
    <div
      className={cn(
        "relative shrink-0 overflow-hidden bg-secondary ring-1 ring-inset ring-white/10",
        SIZE_CLASSES[size],
        className,
      )}
      style={posterUrl ? undefined : { backgroundColor: TONES[hash(title) % TONES.length] }}
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
        <div className="absolute inset-0 flex flex-col justify-end">
          <span className={cn("font-heading text-foreground/90 line-clamp-4", TITLE_CLASSES[size])}>
            {title}
            {year && size !== "sm" && (
              <span className="block font-sans text-[0.7em] text-foreground/50 mt-0.5">{year}</span>
            )}
          </span>
        </div>
      )}
    </div>
  );
}
