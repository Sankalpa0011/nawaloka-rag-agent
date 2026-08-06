// Renders the small markdown subset the assistant produces (bold, italic,
// inline code, lists, headings, paragraphs) as React elements - no
// dangerouslySetInnerHTML, so model output can't inject markup.

import React from "react";

// Order matters: `**bold**` must be tried before `*italic*`.
const INLINE = /(`[^`\n]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_[^_\n]+_)/g;

function renderInline(text: string, keyPrefix: string): React.ReactNode[] {
  const parts = text.split(INLINE).filter((p) => p !== "" && p !== undefined);

  return parts.map((part, i) => {
    const key = `${keyPrefix}-${i}`;

    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={key}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("__") && part.endsWith("__") && part.length > 4) {
      return <strong key={key}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return <code key={key}>{part.slice(1, -1)}</code>;
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
      return <em key={key}>{part.slice(1, -1)}</em>;
    }
    if (part.startsWith("_") && part.endsWith("_") && part.length > 2) {
      return <em key={key}>{part.slice(1, -1)}</em>;
    }
    return <React.Fragment key={key}>{part}</React.Fragment>;
  });
}

const BULLET = /^\s*[-*•]\s+/;
const NUMBERED = /^\s*\d+[.)]\s+/;
const HEADING = /^(#{1,4})\s+(.*)$/;

export function Markdown({ text }: { text: string }) {
  if (!text) return null;

  // Blank lines separate blocks; a run of list lines is also its own block.
  const blocks = text.trim().split(/\n{2,}/);

  return (
    <div className="md">
      {blocks.map((block, bi) => {
        const lines = block.split("\n").filter((l) => l.trim() !== "");
        if (lines.length === 0) return null;

        const heading = lines[0].match(HEADING);
        if (heading && lines.length === 1) {
          const level = Math.min(heading[1].length + 2, 6);
          const Tag = `h${level}` as keyof React.JSX.IntrinsicElements;
          return <Tag key={bi}>{renderInline(heading[2], `h${bi}`)}</Tag>;
        }

        if (lines.every((l) => BULLET.test(l))) {
          return (
            <ul key={bi}>
              {lines.map((l, li) => (
                <li key={li}>{renderInline(l.replace(BULLET, ""), `${bi}-${li}`)}</li>
              ))}
            </ul>
          );
        }

        if (lines.every((l) => NUMBERED.test(l))) {
          return (
            <ol key={bi}>
              {lines.map((l, li) => (
                <li key={li}>{renderInline(l.replace(NUMBERED, ""), `${bi}-${li}`)}</li>
              ))}
            </ol>
          );
        }

        // Mixed block: keep intro text and list items in document order.
        if (lines.some((l) => BULLET.test(l) || NUMBERED.test(l))) {
          const nodes: React.ReactNode[] = [];
          let bucket: string[] = [];

          const flush = (key: string) => {
            if (!bucket.length) return;
            nodes.push(
              <ul key={key}>
                {bucket.map((l, li) => (
                  <li key={li}>
                    {renderInline(l.replace(BULLET, "").replace(NUMBERED, ""), `${key}-${li}`)}
                  </li>
                ))}
              </ul>
            );
            bucket = [];
          };

          lines.forEach((line, li) => {
            if (BULLET.test(line) || NUMBERED.test(line)) {
              bucket.push(line);
            } else {
              flush(`${bi}-l${li}`);
              nodes.push(<p key={`${bi}-p${li}`}>{renderInline(line, `${bi}-${li}`)}</p>);
            }
          });
          flush(`${bi}-tail`);
          return <React.Fragment key={bi}>{nodes}</React.Fragment>;
        }

        // Plain paragraph; single newlines inside it become line breaks.
        return (
          <p key={bi}>
            {lines.map((l, li) => (
              <React.Fragment key={li}>
                {li > 0 && <br />}
                {renderInline(l, `${bi}-${li}`)}
              </React.Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}
