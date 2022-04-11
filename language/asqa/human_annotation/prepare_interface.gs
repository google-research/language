// Enter ID of the setup spreadsheet instead of "TODO"
// Refer to https://stackoverflow.com/questions/36061433/how-to-do-i-locate-a-google-spreadsheet-id
// for instructions on how to locate the ID
var setupDocId = "TODO";

// Enter ID of the **folder** where you want an annotation spreadsheet to be created
var parentFolderId = "TODO";

// Columns with questions/answers int the setup doc
var question_cols = ["G", "H", "I", "J", "K", "L"];
var answers_cols = ["M", "N", "O", "P", "Q", "R"];

// Bold text style
var bold = SpreadsheetApp.newTextStyle()
  .setBold(true)
  .build();

// Questions to ask in the Evaluation 2
var ambiguity_question = "Which paragraph is better in terms of resolving ambiguity?"
var fluency_question = "Which paragraph is better in terms of fluency?"
var overall_question = "Which paragraph is better overall?"

var eval2_questions = [ambiguity_question, fluency_question, overall_question];

function createSheet(name, folderId) {
//This function created a spreadsheet with a given name in a given folder.
//Returns: ID of the created file

  var resource = {
  title: name,
  mimeType: MimeType.GOOGLE_SHEETS,
  parents: [{id: folderId}]
  }
  var fileJson = Drive.Files.insert(resource);
  return(fileJson.id);
}

function buildRichQuestion(text) {
//This function builds the Rich Text representation of the ambiguous question

  var richQuestion = SpreadsheetApp.newRichTextValue()
  .setText("Ambiguous Question: " + text)
  .setTextStyle(0, 20, bold)
  .build();

  return(richQuestion)
}

function workhorse(destination, range, text, align_h, align_v, boldness, fontsize, is_rich, border) {
// This function writes given "text" in the specified "range" of a given "spreadsheet".
// Multiple parameters of the text can be specified:
// align_h -- horizontal alignment of the text
// align_v -- vertical alignment of the text
// boldness -- font weight of the text
// fontsize --  fontsize of the text
// is_rich -- a binary indicator of whether "text" is a simple string or a rich text representation
// border -- locations of borders that should be colored
  tmp = destination.getRange(range)
    .merge()
    .setHorizontalAlignment(align_h)
    .setVerticalAlignment(align_v)
    .setWrapStrategy(SpreadsheetApp.WrapStrategy.WRAP)
    .setFontWeight(boldness)
    .setFontSize(fontsize);

  if(border=="standard") {
    tmp.setBorder(null, null, true, true, null, null);
  } else if(border=="right") {
    tmp.setBorder(null, null, null, true, null, null);
  } else if(border=="full") {
    tmp.setBorder(true, true, true, true, null, null);
  }

  if (is_rich) {
    tmp.setRichTextValue(text);
  } else {
    tmp.setValue(text);
  }

}

function workhorse_checkbox(destination, range, border) {
// This function places checkbox in the specified "range" of a given "spreadsheet".
// The "border" argument specifies the locations of borders that should be colored

  tmp = destination.getRange(range)
  .merge()
  .setHorizontalAlignment("center")
  .setVerticalAlignment("top")
  .insertCheckboxes()
  .setWrapStrategy(SpreadsheetApp.WrapStrategy.WRAP);

  if(border=="standard") {
    tmp.setBorder(null, null, true, true, null, null);
  } else if(border=="right") {
    tmp.setBorder(null, null, null, true, null, null);
  }

}


function main() {
// Main Function

  //Create an annotation spreadsheet, open it, and insert many rows
  var dstId = createSheet("AnnotationInterface", parentFolderId);
  var dst = SpreadsheetApp.openById(dstId).getSheets()[0];
  dst.insertRows(1, 2000);

  //Open the source spreadsheet
  var src = SpreadsheetApp.openById(setupDocId).getSheets()[0];

  //Set column widths in the destination spreadsheet
  dst.setColumnWidth(1, 480);
  dst.setColumnWidth(2, 80);
  dst.setColumnWidth(3, 80);
  dst.setColumnWidth(4, 80);
  dst.setColumnWidth(5, 80);
  dst.setColumnWidth(6, 80);
  dst.setColumnWidth(7, 80);

  // Technical Variables
  var offset = 1;
  var current_col;
  var pointer = 0;
  var k = 2;

  //For each line in the setup spreadsheet, create a block for the corresponding pairwise comparison
  while((src.getRange('A' + k).isBlank() == false)) {

  range = "A" + offset + ":G" + offset;
  text = text='PAIR ' + (k-1);
  workhorse(destination=dst, range=range, text=text, align_h="center", align_v="middle",
        boldness="bold", fontsize=12, is_rich=false, border="standard");

  offset++;

  range = "A" + offset + ":G" + offset;
  text = buildRichQuestion(src.getRange("F" + k).getDisplayValue());
  workhorse(destination=dst, range=range, text=text, align_h="center", align_v="middle",
        boldness="normal", fontsize=12, is_rich=true, border="standard");

  offset++;

  range = "A" + offset;
  text = "Paragraph 1";
  workhorse(destination=dst, range=range, text=text, align_h="center", align_v="middle",
        boldness="bold", fontsize=10, is_rich=false, border="standard");

  range = "B" + offset + ":G" + offset;
  text = "Paragraph 2";
  workhorse(destination=dst, range=range, text=text, align_h="center", align_v="middle",
        boldness="bold", fontsize=10, is_rich=false, border="standard");

  offset++;

  range = "A" + offset;
  text = src.getRange("D" + k).getDisplayValue().toLowerCase();
  workhorse(destination=dst, range=range, text=text, align_h="left", align_v="top",
        boldness="normal", fontsize=10, is_rich=false, border="standard");


  range = "B" + offset + ":G" + offset;
  text = src.getRange("E" + k).getDisplayValue().toLowerCase();
  workhorse(destination=dst, range=range, text=text, align_h="left", align_v="top",
        boldness="normal", fontsize=10, is_rich=false, border="standard");

  offset++;

  range = "A" + offset + ":G" + offset;
  text = "Evaluation 1";
  workhorse(destination=dst, range=range, text=text, align_h="center", align_v="middle",
        boldness="bold", fontsize=10, is_rich=false, border="full");

  offset++;

  range = "A" + offset;
  text = "Disambiguated Questions";
  workhorse(destination=dst, range=range, text=text, align_h="center", align_v="middle",
        boldness="bold", fontsize=10, is_rich=false, border="standard");


  range = "B" + offset + ":D" + offset;
  text = "Paragraph 1";
  workhorse(destination=dst, range=range, text=text, align_h="center", align_v="middle",
        boldness="bold", fontsize=10, is_rich=false, border="standard");


  range = "E" + offset + ":G" + offset;
  text = "Paragraph 2";
  workhorse(destination=dst, range=range, text=text, align_h="center", align_v="middle",
        boldness="bold", fontsize=10, is_rich=false, border="standard");

  offset++;

  // Block of disambiguated questions

  current_col = 0;

  while((current_col < 6) && (src.getRange(question_cols[current_col] + k).getDisplayValue() != 'NA')) {
    var question = "Q:" + src.getRange(question_cols[current_col] + k).getDisplayValue();
    var answers = "A: " + src.getRange(answers_cols[current_col] + k).getDisplayValue();

    range = "A" + offset;
    text = question + "\n" + answers;
    workhorse(destination=dst, range=range, text=text, align_h="left", align_v="top",
        boldness="normal", fontsize=10, is_rich=false, border="right");

    workhorse_checkbox(dst, "B" + offset + ":D" + offset, border="right");
    workhorse_checkbox(dst, "E" + offset + ":G" + offset, border="right");

    current_col ++;
    offset ++;
  }

  range = "A" + offset + ":G" + offset;
  text = "Evaluation 2";
  workhorse(destination=dst, range=range, text=text, align_h="center", align_v="middle",
        boldness="bold", fontsize=10, is_rich=false, border="full");

  offset++;

  for (var question_idx = 0; question_idx < 3; question_idx++) {
    range = "A" + offset + ":A" + (offset + 1);
    text = eval2_questions[question_idx];
    workhorse(destination=dst, range=range, text=text, align_h="left", align_v="middle",
        boldness="normal", fontsize=10, is_rich=false, border="standard");

    range = "B" + offset + ":C" + offset;
    text = "Tie"
    workhorse(destination=dst, range=range, text=text, align_h="center", align_v="middle",
        boldness="bold", fontsize=10, is_rich=false, border="right");

    range = "D" + offset + ":E" + offset;
    text = "Paragraph 1"
    workhorse(destination=dst, range=range, text=text, align_h="center", align_v="middle",
        boldness="bold", fontsize=10, is_rich=false, border="right");

    range = "F" + offset + ":G" + offset;
    text = "Paragraph 2"
    workhorse(destination=dst, range=range, text=text, align_h="center", align_v="middle",
        boldness="bold", fontsize=10, is_rich=false, border="right");

    offset ++;

    workhorse_checkbox(dst, "B" + offset + ":C" + offset, border="standard");
    workhorse_checkbox(dst, "D" + offset + ":E" + offset, border="standard");
    workhorse_checkbox(dst, "F" + offset + ":G" + offset, border="standard");

    offset ++;

  }

  k++
  }
}


