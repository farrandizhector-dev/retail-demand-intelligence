type Variant = 'success' | 'warning' | 'danger' | 'info' | 'neutral'

const variants: Record<Variant, string> = {
  success: 'bg-success-subtle text-success',
  warning: 'bg-warning-subtle text-warning',
  danger: 'bg-danger-subtle text-danger',
  info: 'bg-primary-subtle text-primary',
  neutral: 'bg-background-elevated text-text-secondary',
}

export function Badge({ label, variant = 'neutral' }: { label: string; variant?: Variant }) {
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-badge text-xs font-medium ${variants[variant]}`}>
      {label}
    </span>
  )
}
