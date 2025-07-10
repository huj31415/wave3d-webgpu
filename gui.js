HTMLElement.prototype._appendChild = HTMLElement.prototype.appendChild;
HTMLElement.prototype.appendChild = function (e) {
  this.append(" ");
  return this._appendChild(e);
}

class GUI {
  io = {};
  groups = {};
  parent;
  toggle;

  constructor(title = "", appendAfter) {
    const parentDiv = document.createElement("div");
    parentDiv.id = "controls"
    parentDiv.className = "right-dark";
    this.parent = this.groups["parent"] = parentDiv;

    const toggleBtn = document.createElement("button");
    toggleBtn.id = "toggleSettings";
    toggleBtn.innerText = ">";
    this.toggle = toggleBtn;

    const titleElem = document.createElement("h2");
    titleElem.innerText = title;
    parentDiv.appendChild(titleElem);

    if (appendAfter) {
      appendAfter.after(parentDiv);
      appendAfter.after(toggleBtn);
    } else {
      document.body.appendChild(toggleBtn);
      document.body.appendChild(parentDiv);
    }

    toggleBtn.addEventListener("click", () => {
      toggleBtn.innerText = toggleBtn.innerText === ">" ? "<" : ">";
      parentDiv.classList.toggle("hidden");
      toggleBtn.classList.toggle("inactive");
    });
  }

  addGroup(id, title, html = "") {
    const group = document.createElement("div");
    group.className = "control-group";
    group.id = id;

    const header = document.createElement("b");
    header.innerText = title;

    group.appendChild(header);
    group.appendChild(document.createElement("br"));
    group.innerHTML += html;
    this.parent.appendChild(group);
    this.parent.appendChild(document.createElement("hr"));
    this.groups[id] = group;
    return group;
  }

  addHalfWidthGroups(id1, id2, group = "parent") {
    const group1 = document.createElement("span");
    group1.className = "halfWidth";
    group1.id = id1;
    this.groups[group].appendChild(group1);
    this.groups[id1] = group1;

    const group2 = document.createElement("span");
    group2.className = "halfWidth";
    group2.id = id1;
    this.groups[group].appendChild(group2);
    this.groups[id2] = group2;
  }

  addStringOutput(id, label = "", suffix = "", group = "parent") {
    const span = document.createElement("span");
    span.id = id;

    this.groups[group].append(label + ": ");
    this.groups[group].appendChild(span);
    this.groups[group].append(" " + suffix);
    this.groups[group].appendChild(document.createElement("br"));

    this.io[id] = (val) => (span.innerText = val);
  }

  addNumericOutput(id, label = "", suffix = "", floatPrecision = 2, group = "parent") {
    // const line = document.createElement("div");
    const span = document.createElement("span");
    span.id = id;

    this.groups[group].append(label + ": ");
    this.groups[group].appendChild(span);
    this.groups[group].append(" " + suffix);
    this.groups[group].appendChild(document.createElement("br"));

    this.io[id] = (val) => (span.innerText = floatPrecision == 0 ? parseInt(val) : parseFloat(val).toFixed(floatPrecision));
  }

  addNDimensionalOutput(ids, label = "", suffix = "", separator = ", ", floatPrecision = 2, group = "parent") {
    this.groups[group].append(label + ": ");

    const spans = ids.map(_ => document.createElement("span"));
    ids.forEach((id, i) => {
      spans[i].id = id;
      this.io[id] = (val) => (spans[i].innerText = floatPrecision == 0 ? parseInt(val) : parseFloat(val).toFixed(floatPrecision));
      this.groups[group].appendChild(spans[i]);
      this.groups[group].append(" " + suffix + (i == ids.length - 1 ? "" : separator));
    });
    this.groups[group].appendChild(document.createElement("br"));
  }

  addNumericInput(id, range = true, label, min, max, step, value = (min + max) / 2, floatPrecision = 2, group = "parent", oninput) {
    const input = document.createElement("input");
    input.type = range ? "range" : "number";
    input.id = id;
    input.min = min;
    input.max = max;
    input.step = step;
    input.value = value;

    const valueSpan = document.createElement("span");
    valueSpan.id = id + "Value";
    valueSpan.innerText = floatPrecision == 0 ? parseInt(value) : parseFloat(value).toFixed(2);

    const labelEl = document.createElement("label");
    labelEl.setAttribute("for", id);
    labelEl.innerText = `${label}: `;
    labelEl.appendChild(valueSpan);

    this.groups[group].appendChild(input);
    this.groups[group].appendChild(labelEl);
    this.groups[group].appendChild(document.createElement("br"));

    this.io[id] = input;
    // this.io[id + "Value"] = valueSpan;

    input.addEventListener("input", () => {
      valueSpan.innerText = floatPrecision == 0 ? parseInt(input.value) : parseFloat(input.value).toFixed(floatPrecision);
      if (oninput) oninput(input.value);
    });
  }

  addCheckbox(id, label, startChecked = false, group = "parent", onclick) {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = id;
    input.checked = startChecked;

    const labelEl = document.createElement("label");
    labelEl.setAttribute("for", id);
    labelEl.innerText = label;

    this.groups[group].appendChild(input);
    this.groups[group].appendChild(labelEl);
    this.groups[group].appendChild(document.createElement("br"));

    this.io[id] = input;

    if (onclick != null) input.addEventListener("click", () => onclick(input.checked));
  }

  addDropdown(id, label, options = [], group = "parent", visibilityMap = {}) {
    const labelEl = document.createElement("label");
    labelEl.setAttribute("for", id);
    labelEl.innerText = label + ": ";

    const select = document.createElement("select");
    select.id = id;

    options.forEach(opt => {
      const option = document.createElement("option");
      option.value = opt;
      option.text = opt;
      select.appendChild(option);
    });

    labelEl.appendChild(select);
    this.groups[group].insertBefore(select, this.io[Object.entries(visibilityMap)[0]]);
    select.after(document.createElement("br"));

    this.io[id] = select;

    // Visibility logic for additional inputs
    select.addEventListener("change", () => {
      Object.entries(visibilityMap).forEach(([val, elements]) => {
        elements.forEach(el => {
          if (typeof el === 'string') el = this.io[el];
          if (el) el.style.display = (select.value === val) ? "" : "none";
        });
      });
    });

    // Initialize visibility based on default
    select.dispatchEvent(new Event("change"));
  }

  addButton(id, label, fullWidth = false, group = "parent", onclick) {
    const button = document.createElement("button");
    if (fullWidth) button.classList.add("fullwidth");
    button.id = id;
    button.innerText = label;
    button.addEventListener("click", onclick);

    this.groups[group].appendChild(button);

    this.io[id] = button;
  }

  addRadioOptions(name, options = [], defaultValue = null, group = "parent") {
    const container = document.createElement("div");

    options.forEach((value) => {
      const input = document.createElement("input");
      input.type = "radio";
      input.name = name;
      input.value = value;
      input.id = `${name}_${value}`;
      if (value === defaultValue) input.checked = true;

      const labelEl = document.createElement("label");
      labelEl.setAttribute("for", input.id);
      // labelEl.innerText = label;
      labelEl.innerText = value;

      container.appendChild(input);
      container.appendChild(labelEl);
      container.appendChild(document.createElement("br"));
    });

    this.groups[group].appendChild(container);

    this.io[name] = () => {
      const selected = container.querySelector(`input[name="${name}"]:checked`);
      return selected ? selected.value : null;
    };
  }
}
