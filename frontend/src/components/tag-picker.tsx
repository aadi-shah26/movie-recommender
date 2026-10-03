import { useState } from "react";
import { Tag, Plus } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { PRESET_TAGS } from "@/lib/storage/favorites";

interface TagPickerProps {
  selected: string[];
  customTags: string[];
  onChange: (tags: string[]) => void;
  onAddCustomTag: (tag: string) => void;
}

export function TagPicker({ selected, customTags, onChange, onAddCustomTag }: TagPickerProps) {
  const [draft, setDraft] = useState("");
  const all = Array.from(new Set([...PRESET_TAGS, ...customTags]));

  const toggle = (tag: string) => {
    if (selected.includes(tag)) onChange(selected.filter((t) => t !== tag));
    else onChange([...selected, tag]);
  };

  const handleAdd = () => {
    const t = draft.trim();
    if (!t) return;
    onAddCustomTag(t);
    if (!selected.includes(t)) onChange([...selected, t]);
    setDraft("");
  };

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          onClick={(e) => e.stopPropagation()}
          className="inline-flex items-center gap-1 rounded-full border border-border/60 bg-secondary/40 hover:bg-secondary px-2 py-0.5 text-[10px] font-medium text-muted-foreground transition-colors"
        >
          <Tag className="h-3 w-3" />
          Tags
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-60 p-3" align="end" onClick={(e) => e.stopPropagation()}>
        <p className="text-xs font-semibold mb-2 text-muted-foreground uppercase tracking-wide">
          Tags
        </p>
        <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
          {all.map((tag) => (
            <label
              key={tag}
              className="flex items-center gap-2 text-sm cursor-pointer hover:bg-secondary/50 rounded px-1.5 py-1"
            >
              <Checkbox checked={selected.includes(tag)} onCheckedChange={() => toggle(tag)} />
              <span className="truncate">{tag}</span>
            </label>
          ))}
        </div>
        <div className="mt-3 flex gap-1.5">
          <Input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleAdd();
              }
            }}
            placeholder="New tag"
            className="h-8 text-xs"
          />
          <Button size="sm" variant="secondary" className="h-8 px-2" onClick={handleAdd}>
            <Plus className="h-3.5 w-3.5" />
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
