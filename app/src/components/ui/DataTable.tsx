import { ReactNode } from 'react'

interface DataTableProps {
  headers: string[]
  rows: (string | number | ReactNode)[][]
  highlightRow?: number
}

export function DataTable({ headers, rows, highlightRow }: DataTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="bg-background-elevated">
            {headers.map((h, i) => (
              <th key={i} className="text-left px-4 py-3 text-xs uppercase tracking-wider text-text-tertiary font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr
              key={ri}
              className={`border-b border-border-subtle last:border-0 transition-colors hover:bg-background-hover ${
                ri % 2 === 1 ? 'bg-background-base/30' : ''
              } ${highlightRow === ri ? 'bg-primary-subtle' : ''}`}
            >
              {row.map((cell, ci) => (
                <td key={ci} className="px-4 py-3 text-sm text-text-secondary">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
