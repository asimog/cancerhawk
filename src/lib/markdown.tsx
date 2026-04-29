import React from 'react';

function inline(value: string) {
  const parts = value.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={index}>{part.slice(1, -1)}</code>;
    }
    return <React.Fragment key={index}>{part}</React.Fragment>;
  });
}

function splitTableRow(line: string) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim());
}

function isSeparator(line: string) {
  return splitTableRow(line).every((cell) => /^:?-{3,}:?$/.test(cell));
}

export function Markdown({ content, skipTitle = false }: { content: string; skipTitle?: boolean }) {
  const lines = content.split('\n');
  const blocks: React.ReactNode[] = [];
  let paragraph: string[] = [];

  function flushParagraph() {
    if (!paragraph.length) return;
    blocks.push(<p key={blocks.length}>{inline(paragraph.join(' '))}</p>);
    paragraph = [];
  }

  for (let i = 0; i < lines.length; i += 1) {
    const raw = lines[i];
    const line = raw.trim();
    if (!line) {
      flushParagraph();
      continue;
    }
    if (skipTitle && line.startsWith('# ')) continue;
    if (line.startsWith('## ')) {
      flushParagraph();
      blocks.push(<h2 key={blocks.length}>{line.slice(3)}</h2>);
      continue;
    }
    if (line.startsWith('# ')) {
      flushParagraph();
      blocks.push(<h1 key={blocks.length}>{line.slice(2)}</h1>);
      continue;
    }
    if (line.startsWith('|') && lines[i + 1]?.trim().startsWith('|') && isSeparator(lines[i + 1].trim())) {
      flushParagraph();
      const headers = splitTableRow(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        rows.push(splitTableRow(lines[i]));
        i += 1;
      }
      i -= 1;
      blocks.push(
        <div className="table-scroll" key={blocks.length}>
          <table>
            <thead>
              <tr>{headers.map((header) => <th key={header}>{inline(header)}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {headers.map((_, cellIndex) => <td key={cellIndex}>{inline(row[cellIndex] || '')}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }
    if (/^\d+\.\s+/.test(line)) {
      flushParagraph();
      const items: string[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ''));
        i += 1;
      }
      i -= 1;
      blocks.push(<ol key={blocks.length}>{items.map((item) => <li key={item}>{inline(item)}</li>)}</ol>);
      continue;
    }
    if (line.startsWith('- ')) {
      flushParagraph();
      const items: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith('- ')) {
        items.push(lines[i].trim().slice(2));
        i += 1;
      }
      i -= 1;
      blocks.push(<ul key={blocks.length}>{items.map((item) => <li key={item}>{inline(item)}</li>)}</ul>);
      continue;
    }
    paragraph.push(line);
  }
  flushParagraph();
  return <>{blocks}</>;
}
