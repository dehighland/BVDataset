let slider = document.getElementById("myRange");
let redRect = document.querySelector(".rectangle.red");
let greenRect = document.querySelector(".rectangle.green");

function updateRectangles() {
  let width = slider.value;
  redRect.style.width = `calc(${width}% - 10px)`;
  greenRect.style.width = `calc(${100 - width}% - 10px)`;
}

updateRectangles();

slider.oninput = function() {
  updateRectangles();
}