var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function randImage() {
  window.event.preventDefault();
  const imgArray = ['a01-122.png', 'a03-014.png', 'b04-162.png', 'f04-093.png', 'g06-042j.png', 'j01-045.png',
                    'l01-173.png', 'l03-004.png', 'p02-105.png', 'p03-040.png', 'r02-081.png'];

  // subtract current image from imgArray before selecting so as not to randomly select current image
  var curImg = el("upload-label").innerHTML;
  var idx = imgArray.indexOf(curImg);
  if (idx > -1) {
    imgArray.splice(idx, 1);
  }

  var num = Math.floor( Math.random() * imgArray.length );
  var img = imgArray[ num ];

  el("upload-label").innerHTML = img;
  el("upload-label").className = "";
  el("image-picked").src = 'static/images/'+img;
  el("image-picked").className = "";

  el('test-label').className = "";
  // empty the FileList object associated with 'file-input'
  el("file-input").value = ''
}

function showPicked(input) {
  el('test-label').className = "no-display";

  el("upload-label").innerHTML = input.files[0].name;
  el("upload-label").className = "";
  var reader = new FileReader();
  reader.onload = function(e) {
    el("image-picked").src = e.target.result;
    el("image-picked").className = "";
  };
  reader.readAsDataURL(input.files[0]);
}

function analyze() {
  var fileData = new FormData();

  var uploadFiles = el("file-input").files;    //check for uploaded image
  var fileName = el("upload-label").innerHTML; //check for chosen filename

  if (uploadFiles.length === 1) {
    fileData.append("file", uploadFiles[0]);
  } else if (fileName !== "No file chosen") {
    fileData.append("filename", fileName);
  } else {
    alert("Please select a file to analyze!");
    return;
  }

  el("analyze-button").innerHTML = "Analyzing...";
  el("result-label").innerHTML = "<em>Please be patient.<br>This may take up to 20 seconds.<br>Reading is hard :/</em>";
  el("spinner").className = "";
  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`, true);
  xhr.onerror = function() {
    alert(xhr.responseText);
  };
  xhr.onload = function(e) {
    if (this.readyState === 4) {
      var response = JSON.parse(e.target.responseText);
      el("result-label").innerHTML = response["result"];
    }
    el("analyze-button").innerHTML = "Analyze";
    el("spinner").className = "no-display";
  };

  xhr.send(fileData);
}