function setMenuControls() {
  // Shift nav in mobile when clicking the menu.
  let toggles = document.querySelectorAll("[data-toggle='wy-nav-top']");
  for (let t of toggles) {
    t.onclick = function(){
      for (let e of document.querySelectorAll("[data-toggle='wy-nav-shift']")){
	e.classList.toggle('shift');
      }
    };
  }

  // Close menu when you click a link.
  let links = document.querySelectorAll(".wy-menu-vertical .current ul li a");
  for (let l of links) {
    l.onclick = function(){
      for (let e of document.querySelectorAll("[data-toggle='wy-nav-shift']")){
	e.classList.toggle('shift');
      }
    };
  }
}

document.addEventListener("DOMContentLoaded", (event) => {
  setMenuControls();
});
