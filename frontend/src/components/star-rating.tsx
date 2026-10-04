import { Star } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

interface StarRatingProps {
  value: number;
  onChange?: (v: number) => void;
  size?: number;
  readOnly?: boolean;
}

export function StarRating({ value, onChange, size = 14, readOnly }: StarRatingProps) {
  const [hover, setHover] = useState(0);
  const display = hover || value;

  return (
    <div className="inline-flex items-center gap-0.5" onMouseLeave={() => setHover(0)}>
      {[1, 2, 3, 4, 5].map((i) => {
        const active = i <= display;
        return (
          <button
            key={i}
            type="button"
            disabled={readOnly}
            onMouseEnter={() => !readOnly && setHover(i)}
            onClick={(e) => {
              e.stopPropagation();
              if (readOnly) return;
              onChange?.(value === i ? 0 : i);
            }}
            className={cn("p-0.5", readOnly ? "cursor-default" : "cursor-pointer")}
            aria-label={`Rate ${i} star${i > 1 ? "s" : ""}`}
          >
            <Star
              size={size}
              className={cn(
                "transition-colors",
                active ? "fill-primary text-primary" : "text-muted-foreground/35",
              )}
            />
          </button>
        );
      })}
    </div>
  );
}
