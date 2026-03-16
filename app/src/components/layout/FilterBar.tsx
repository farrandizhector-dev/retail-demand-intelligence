interface FilterBarProps {
  state: string
  onState: (v: string) => void
  category: string
  onCategory: (v: string) => void
}

export function FilterBar({ state, onState, category, onCategory }: FilterBarProps) {
  const selectClass = "h-9 bg-background-elevated border border-border-DEFAULT rounded-button text-sm text-text-secondary px-3 focus:ring-1 focus:ring-primary focus:outline-none cursor-pointer"
  return (
    <div className="flex items-center gap-3">
      <select value={state} onChange={e => onState(e.target.value)} className={selectClass}>
        <option value="">All States</option>
        <option value="CA">CA</option>
        <option value="TX">TX</option>
        <option value="WI">WI</option>
      </select>
      <select value={category} onChange={e => onCategory(e.target.value)} className={selectClass}>
        <option value="">All Categories</option>
        <option value="FOODS">FOODS</option>
        <option value="HOUSEHOLD">HOUSEHOLD</option>
        <option value="HOBBIES">HOBBIES</option>
      </select>
    </div>
  )
}
