//! Shared data for report sections that display a table.

// Rust guideline compliant 2026-10-09

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::report_helpers::escape_html;

/// A titled table with an explanatory note and ordered rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportTable {
    /// Section heading.
    pub title: String,
    /// Text shown before the table.
    pub note: String,
    /// Column labels in display order.
    pub columns: Vec<String>,
    /// Display values in column order.
    pub rows: Vec<Vec<String>>,
}

/// A heading and the ordered tables that belong under it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportTableGroup {
    /// Group heading.
    pub title: String,
    /// Tables in display order.
    pub tables: Vec<ReportTable>,
}

pub(crate) fn render_table_html(table: &ReportTable) -> String {
    let mut html = String::new();
    let _ = write!(
        html,
        "<div class=\"plot-container\"><h2>{}</h2><p class=\"epa-footer\">{}</p>",
        escape_html(&table.title),
        escape_html(&table.note),
    );
    write_table_html(&mut html, table, "bm-table", "");
    html.push_str("</div>");
    html
}

pub(crate) fn write_table_html(
    html: &mut String,
    table: &ReportTable,
    class: &str,
    row_break: &str,
) {
    if class.is_empty() {
        html.push_str("<table>");
    } else {
        let _ = write!(html, "<table class=\"{class}\">");
    }
    html.push_str("<thead><tr>");
    for column in &table.columns {
        let _ = write!(html, "<th>{}</th>", escape_html(column));
    }
    html.push_str("</tr></thead><tbody>");
    html.push_str(row_break);
    for row in &table.rows {
        html.push_str("<tr>");
        for cell in row {
            let _ = write!(html, "<td>{}</td>", escape_html(cell));
        }
        html.push_str("</tr>");
        html.push_str(row_break);
    }
    html.push_str("</tbody></table>");
}
