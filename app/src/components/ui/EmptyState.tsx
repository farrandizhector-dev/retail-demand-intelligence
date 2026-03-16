import { ReactNode } from 'react'

export function EmptyState({ icon, title, description }: { icon: ReactNode; title: string; description?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16">
      <div className="text-text-tertiary mb-4" style={{ fontSize: 48 }}>{icon}</div>
      <div className="text-sm font-medium text-text-secondary mb-1">{title}</div>
      {description && <div className="text-xs text-text-tertiary max-w-sm text-center">{description}</div>}
    </div>
  )
}
