'use client';

import { ReactNode } from 'react';
import { TypeTable } from 'fumadocs-ui/components/type-table';

interface AttributeInfo {
  type: string;
  description: string;
  readonly?: boolean;
  default?: string;
}

interface PythonClassProps {
  name: string;
  description: string;
  bases?: string[];
  attributes?: Record<string, AttributeInfo>;
  source?: string;
  children?: ReactNode;
}

export function PythonClass({
  name,
  description,
  bases,
  attributes,
  source,
  children,
}: PythonClassProps) {
  // Convert attributes to TypeTable format
  const attributeTableProps: Record<string, {
    type: string;
    description: string;
    default?: string;
  }> = {};

  if (attributes) {
    for (const [attrName, attrInfo] of Object.entries(attributes)) {
      const desc = attrInfo.readonly 
        ? `${attrInfo.description} (read-only)`
        : attrInfo.description;
      
      attributeTableProps[attrName] = {
        type: attrInfo.type,
        description: desc,
        default: attrInfo.default,
      };
    }
  }

  return (
    <div className="my-8 rounded-lg border-2 border-fd-primary/20 bg-fd-card">
      {/* Class header */}
      <div className="border-b border-fd-border bg-fd-muted/50 p-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="font-mono text-xl font-bold text-fd-foreground m-0">
              class {name}
              {bases && bases.length > 0 && (
                <span className="text-fd-muted-foreground font-normal">
                  ({bases.join(', ')})
                </span>
              )}
            </h2>
          </div>
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
        <p className="mt-3 text-fd-muted-foreground mb-0">{description}</p>
      </div>

      {/* Class body */}
      <div className="p-4">
        {/* Attributes section */}
        {attributes && Object.keys(attributes).length > 0 && (
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-fd-foreground mb-3">
              Attributes
            </h3>
            <TypeTable type={attributeTableProps} />
          </div>
        )}

        {/* Methods section */}
        {children && (
          <div>
            <h3 className="text-lg font-semibold text-fd-foreground mb-3">
              Methods
            </h3>
            <div className="space-y-4">
              {children}
            </div>
          </div>
        )}

        {/* Empty state if no attributes or methods */}
        {!attributes && !children && (
          <p className="text-sm text-fd-muted-foreground italic">
            No public attributes or methods documented.
          </p>
        )}
      </div>
    </div>
  );
}
