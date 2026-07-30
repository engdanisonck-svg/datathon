import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = process.argv[2];
if (!inputPath) throw new Error("Usage: node inspect_workbook.mjs <input.xlsx>");

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const overview = await workbook.inspect({
  kind: "workbook,sheet,table,region",
  maxChars: 18000,
  tableMaxRows: 8,
  tableMaxCols: 40,
  tableMaxCellChars: 120,
});
console.log(overview.ndjson);

for (const sheet of workbook.worksheets.items) {
  const used = sheet.getUsedRange(true);
  if (!used) continue;
  const region = await workbook.inspect({
    kind: "region",
    sheetId: sheet.name,
    range: used.address,
    maxChars: 24000,
    tableMaxRows: 12,
    tableMaxCols: 80,
    tableMaxCellChars: 160,
  });
  console.log(region.ndjson);
}
