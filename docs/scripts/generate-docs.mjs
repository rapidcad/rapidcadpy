import { rimraf } from 'rimraf';
import * as Python from 'fumadocs-python';
import * as fs from 'node:fs/promises';
// output JSON file path
const jsonPath = './rapidcadpy.json';
async function generate() {
  const out = '../content/docs/python';
  // clean previous output
  await rimraf(out);
  const content = JSON.parse((await fs.readFile(jsonPath)).toString());
  const converted = Python.convert(content, {
    baseUrl: '/docs',
  });
  await Python.write(converted, {
    outDir: out,
  });
}
void generate();