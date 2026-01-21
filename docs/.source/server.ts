// @ts-nocheck
import { default as __fd_glob_22 } from "../content/docs/finite-element-analysis/meta.json?collection=docs"
import { default as __fd_glob_21 } from "../content/docs/advanced/meta.json?collection=docs"
import { default as __fd_glob_20 } from "../content/docs/computer-aided-design/meta.json?collection=docs"
import { default as __fd_glob_19 } from "../content/docs/meta.json?collection=docs"
import * as __fd_glob_18 from "../content/docs/finite-element-analysis/visualization.mdx?collection=docs"
import * as __fd_glob_17 from "../content/docs/finite-element-analysis/optimization.mdx?collection=docs"
import * as __fd_glob_16 from "../content/docs/finite-element-analysis/materials.mdx?collection=docs"
import * as __fd_glob_15 from "../content/docs/finite-element-analysis/loads.mdx?collection=docs"
import * as __fd_glob_14 from "../content/docs/finite-element-analysis/fea-analysis.mdx?collection=docs"
import * as __fd_glob_13 from "../content/docs/finite-element-analysis/constraints.mdx?collection=docs"
import * as __fd_glob_12 from "../content/docs/advanced/visualizations.mdx?collection=docs"
import * as __fd_glob_11 from "../content/docs/advanced/inventor-reverse-engineer.mdx?collection=docs"
import * as __fd_glob_10 from "../content/docs/advanced/3d-exports.mdx?collection=docs"
import * as __fd_glob_9 from "../content/docs/advanced/2d-exports.mdx?collection=docs"
import * as __fd_glob_8 from "../content/docs/computer-aided-design/workplanes.mdx?collection=docs"
import * as __fd_glob_7 from "../content/docs/computer-aided-design/sweep.mdx?collection=docs"
import * as __fd_glob_6 from "../content/docs/computer-aided-design/shape.mdx?collection=docs"
import * as __fd_glob_5 from "../content/docs/computer-aided-design/overview.mdx?collection=docs"
import * as __fd_glob_4 from "../content/docs/computer-aided-design/index.mdx?collection=docs"
import * as __fd_glob_3 from "../content/docs/computer-aided-design/fluent-api.mdx?collection=docs"
import * as __fd_glob_2 from "../content/docs/computer-aided-design/built-in-profiles.mdx?collection=docs"
import * as __fd_glob_1 from "../content/docs/api/mesher-base-example.mdx?collection=docs"
import * as __fd_glob_0 from "../content/docs/index.mdx?collection=docs"
import { server } from 'fumadocs-mdx/runtime/server';
import type * as Config from '../source.config';

const create = server<typeof Config, import("fumadocs-mdx/runtime/types").InternalTypeConfig & {
  DocData: {
  }
}>({"doc":{"passthroughs":["extractedReferences"]}});

export const docs = await create.docs("docs", "content/docs", {"meta.json": __fd_glob_19, "computer-aided-design/meta.json": __fd_glob_20, "advanced/meta.json": __fd_glob_21, "finite-element-analysis/meta.json": __fd_glob_22, }, {"index.mdx": __fd_glob_0, "api/mesher-base-example.mdx": __fd_glob_1, "computer-aided-design/built-in-profiles.mdx": __fd_glob_2, "computer-aided-design/fluent-api.mdx": __fd_glob_3, "computer-aided-design/index.mdx": __fd_glob_4, "computer-aided-design/overview.mdx": __fd_glob_5, "computer-aided-design/shape.mdx": __fd_glob_6, "computer-aided-design/sweep.mdx": __fd_glob_7, "computer-aided-design/workplanes.mdx": __fd_glob_8, "advanced/2d-exports.mdx": __fd_glob_9, "advanced/3d-exports.mdx": __fd_glob_10, "advanced/inventor-reverse-engineer.mdx": __fd_glob_11, "advanced/visualizations.mdx": __fd_glob_12, "finite-element-analysis/constraints.mdx": __fd_glob_13, "finite-element-analysis/fea-analysis.mdx": __fd_glob_14, "finite-element-analysis/loads.mdx": __fd_glob_15, "finite-element-analysis/materials.mdx": __fd_glob_16, "finite-element-analysis/optimization.mdx": __fd_glob_17, "finite-element-analysis/visualization.mdx": __fd_glob_18, });