'use client';

import { TypeTable } from 'fumadocs-ui/components/type-table';

interface ParamInfo {
  type: string;
  description: string;
  required?: boolean;
  default?: string;
}

interface ReturnInfo {
  type: string;
  description: string;
}

interface PythonFunctionProps {
  name: string;
  signature: string;
  description: string;
  params: Record<string, ParamInfo>;
  returns?: ReturnInfo;
  source?: string;
}

export function PythonFunction({
  name,
  signature,
  description,
  params,
  returns,
  source,
}: PythonFunctionProps) {
  // Convert params to TypeTable format
  const typeTableProps: Record<string, {
    type: string;
    description: string;
    required?: boolean;
    default?: string;
  }> = {};

  for (const [paramName, paramInfo] of Object.entries(params)) {
    typeTableProps[paramName] = {
      type: paramInfo.type,
      description: paramInfo.description,
      required: paramInfo.required,
      default: paramInfo.default,
    };
  }

  return (
    <div className="my-6 rounded-lg border bg-fd-card p-4">
      <div className="flex items-start justify-between gap-4">
        <h3 className="font-mono text-lg font-semibold text-fd-foreground m-0">
          {signature}
        </h3>
        {source && (
          <a
            href={source}
            className="text-sm text-fd-muted-foreground hover:text-fd-foreground shrink-0"
            target="_blank"
            rel="noopener noreferrer"
          >
            [source]
          </a>
        )}
      </div>

      <p className="mt-3 text-fd-muted-foreground">{description}</p>

      {Object.keys(params).length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-semibold text-fd-foreground mb-2">Parameters</h4>
          <TypeTable type={typeTableProps} />
        </div>
      )}

      {returns && (
        <div className="mt-4 space-y-1">
          <p className="text-sm">
            <span className="font-semibold text-fd-foreground">Returns:</span>{' '}
            <span className="text-fd-muted-foreground">{returns.description}</span>
          </p>
          <p className="text-sm">
            <span className="font-semibold text-fd-foreground">Return type:</span>{' '}
            <code className="rounded bg-fd-muted px-1.5 py-0.5 font-mono text-sm">
              {returns.type}
            </code>
          </p>
        </div>
      )}
    </div>
  );
}
