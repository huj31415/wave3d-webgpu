
class GUI {
  io = {};
  groups = {};

  /**
   * Creates a GUI object
   * @param {String} title Title of the application
   * @param {HTMLElement} appendAfter Element to add the GUI after
   */
  constructor(title = "", appendAfter) {
    const parentDiv = document.createElement("div");
    parentDiv.id = "controls"
    parentDiv.className = "right-dark";
    this.groups["parent"] = parentDiv;

    const toggleBtn = document.createElement("button");
    toggleBtn.id = "toggleSettings";
    toggleBtn.innerText = ">";

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

  /**
   * Adds a group to the GUI
   * @param {String} id id of the group, for adding other IO under it
   * @param {String} title Title of the group
   * @param {String} html Plain HTML to add to to the group
   * @param {String} group ID of the group to add this under
   * @returns The newly created group
   */
  addGroup(id, title, html, group = "parent") {
    const newGroup = document.createElement("div");
    newGroup.className = "control-group";
    newGroup.id = id;

    if (title) {
      const header = document.createElement("b");
      header.innerText = title;
      this.groups[group].appendChild(header);
      header.addEventListener("click", () => {
        newGroup.classList.toggle("hidden");
      });
    }

    if (html) newGroup.innerHTML += html;
    this.groups[group].appendChild(newGroup);
    if (title) this.groups[group].appendChild(document.createElement("hr"));
    this.groups[id] = newGroup;

    return newGroup;
  }

  /**
   * Creates half-width groups under a specified group
   * @param {String} id1 id of the left group
   * @param {String} id2 id of the right group
   * @param {String} group id of the group to add this under
   */
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

  /**
   * Adds a string output to the gui
   * @param {String} id id of the output
   * @param {String} label Label (prefix) of the output value, a colon will be added
   * @param {String} suffix String appended after the output value, eg. units
   * @param {String} group id of the group to add this under
   */
  addStringOutput(id, label = "", suffix = "", group = "parent") {
    const span = document.createElement("span");
    span.id = id;

    this.groups[group].append(label + ": ");
    this.groups[group].appendChild(span);
    this.groups[group].append(" " + suffix);
    this.groups[group].appendChild(document.createElement("br"));

    this.io[id] = (val) => (span.innerText = val);
  }

  /**
   * Adds a numeric output to the gui
   * @param {String} id id of the output
   * @param {String} label Label (prefix) of the output value, a colon will be added
   * @param {String} suffix String appended after the output value, eg. units
   * @param {Number} floatPrecision Precision of the printed value, set to 0 for integer
   * @param {String} group id of the group to add this under
   */
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

  /**
   * Creates a multidimensional numeric output
   * @param {Array<String>} ids Array of ids for each dimension of the output
   * @param {String} label Label of the output value, a colon will be added
   * @param {String} suffix String appended after the output value, eg. units
   * @param {String} separator String used to separate the individual numbers
   * @param {Number} floatPrecision Precision of the printed value, set to 0 for integer
   * @param {String} group id of the group to add this under
   */
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

  /**
   * Adds a numeric input
   * @param {String} id id of the input. A container div is created with id `${id}-container`
   * @param {Boolean} range Whether to use a range input or a plain number input
   * @param {String} label Label of the input, a colon and the value will be added if range is true
   * @param {Number} min Min value
   * @param {Number} max Max value
   * @param {Number} step Step size
   * @param {Number} value Value at initialization
   * @param {Number} floatPrecision Precision of the output display
   * @param {String} group id of the group to add this under
   * @param {Function} oninput Callback function of the new value to run on user input
   * @param {String} title Description as a tooltip
   */
  addNumericInput(id, range = true, label, min, max, step, value = (min + max) / 2, floatPrecision = 2, group = "parent", oninput, title) {
    const container = document.createElement("div");
    container.id = `${id}-container`;

    const input = document.createElement("input");
    input.type = range ? "range" : "number";
    input.id = id;
    input.min = min;
    input.max = max;
    input.step = step;
    input.value = value;

    const labelEl = document.createElement("label");
    labelEl.setAttribute("for", id);
    labelEl.innerText = range ? `${label}: ` : label;
    if (title) labelEl.title = title;

    let valueSpan = document.createElement("span");
    valueSpan.id = id + "Value";
    valueSpan.innerText = floatPrecision == 0 ? parseInt(value) : parseFloat(value).toFixed(floatPrecision);

    if (range) labelEl.appendChild(valueSpan);

    container.appendChild(input);
    container.appendChild(labelEl);

    this.groups[group].appendChild(container);

    this.io[id] = input;

    input.addEventListener("input", () => {
      if (range) valueSpan.innerText = floatPrecision == 0 ? parseInt(input.value) : parseFloat(input.value).toFixed(floatPrecision);
      if (oninput) oninput((floatPrecision == 0 ? parseInt : parseFloat)(input.value));
    });

    labelEl.addEventListener("click", () => {
      range = !range;
      input.type = range ? "range" : "number";
      labelEl.textContent = range ? `${label}: ` : label;
      valueSpan.innerText = floatPrecision == 0 ? parseInt(value) : parseFloat(value).toFixed(floatPrecision);
      if (range) labelEl.appendChild(valueSpan);
    })
  }

  /**
   * Adds a checkbox
   * @param {String} id id of the input. A container div is created with id `${id}-container`
   * @param {String} label Label of the input
   * @param {Boolean} startChecked Whether to initialize checked
   * @param {String} group id of the group to add this under
   * @param {Function} onclick Callback function of checked state to run on user input
   */
  addCheckbox(id, label, startChecked = false, group = "parent", onclick) {
    const container = document.createElement("div");
    container.id = `${id}-container`;

    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = id;
    input.checked = startChecked;

    const labelEl = document.createElement("label");
    labelEl.setAttribute("for", id);
    labelEl.innerText = label;

    container.appendChild(input);
    container.appendChild(labelEl);

    this.groups[group].appendChild(container);

    this.io[id] = input;

    if (onclick != null) input.addEventListener("click", () => onclick(input.checked));
  }

  /**
   * Adds a dropdown selector
   * @param {String} id id of the input. A container div is created with id `${id}-container`
   * @param {String} label Label
   * @param {Array<String>} options Array of string options which are also values
   * @param {String} group id of the group to add this under
   * @param {Object} visibilityMap Object of format `"option": ["id1", "id2"]` of input ids to display when the option is selected
   * @param {Function} onChange Callback function of the currently selected value to run on user input
   */
  addDropdown(id, label, options = [], group = "parent", visibilityMap = {}, onChange) {
    const container = document.createElement("div");
    container.id = `${id}-container`;

    const labelEl = document.createElement("label");
    labelEl.setAttribute("for", id);
    labelEl.innerText = label;

    const select = document.createElement("select");
    select.id = id;

    options.forEach(opt => {
      const option = document.createElement("option");
      option.value = opt;
      option.text = opt;
      select.appendChild(option);
    });

    container.appendChild(select);
    container.appendChild(labelEl);

    const firstInput = document.getElementById(`${Object.entries(visibilityMap)[0][1][0]}-container`);
    this.groups[group].insertBefore(container, firstInput);
    this.io[id] = select;

    // Build reverse visibility map: inputId → [values]
    const inputMap = {};
    Object.entries(visibilityMap).forEach(([val, ids]) => {
      ids.forEach(inputId => {
        if (!inputMap[inputId]) inputMap[inputId] = [];
        inputMap[inputId].push(val);
      });
    });

    const updateVisibility = () => {
      const selected = select.value;
      Object.entries(inputMap).forEach(([inputId, allowedVals]) => {
        const el = document.getElementById(`${inputId}-container`);
        if (el) el.style.display = allowedVals.includes(selected) ? "" : "none";
      });
      if (onChange) onChange(selected);
    };

    select.addEventListener("change", updateVisibility);
    updateVisibility(); // initialize
  }

  /**
   * Adds a button
   * @param {String} id id of the input
   * @param {String} label Label
   * @param {Boolean} fullWidth Whether to add a half or full width button
   * @param {String} group id of the group to add this under
   * @param {Function} onclick Callback function to run when the button is clicked
   */
  addButton(id, label, fullWidth = false, group = "parent", onclick) {
    const button = document.createElement("button");
    if (fullWidth) button.classList.add("fullwidth");
    button.id = id;
    button.innerText = label;
    button.addEventListener("click", onclick);

    this.groups[group].appendChild(button);

    this.io[id] = button;
  }

  /**
 * Adds a set of radio options with optional input visibility mapping
 * @param {String} name name of the input
 * @param {Array<String>} options Array of string options which are also the values
 * @param {String} defaultValue Value to be selected at initialization
 * @param {String} group id of the group to add this under
 * @param {Function} onChange Callback function of the selected value to run on user input
 * @param {Object} visibilityMap Object of format `"option": ["id1", "id2"]` of input ids to display when the option is selected
 */
  addRadioOptions(name, options = [], defaultValue, group = "parent", visibilityMap = {}, onChange) {
    const container = document.createElement("div");
    container.id = `${name}-container`;
    container.classList.add("radioContainer");

    // Build reverse visibility map: inputId -> [allowedRadioValues]
    const inputMap = {};
    Object.entries(visibilityMap).forEach(([val, ids]) => {
      ids.forEach(inputId => {
        if (!inputMap[inputId]) inputMap[inputId] = [];
        inputMap[inputId].push(val);
      });
    });

    const updateVisibility = (selectedValue) => {
      Object.entries(inputMap).forEach(([inputId, allowedVals]) => {
        const el = document.getElementById(`${inputId}-container`);
        if (el) el.style.display = allowedVals.includes(selectedValue) ? "" : "none";
      });
    };


    options.forEach((value) => {
      const input = document.createElement("input");
      input.type = "radio";
      input.name = name;
      input.value = value;
      input.id = `${name}_${value}`;
      if (value === defaultValue) input.checked = true;

      input.addEventListener("change", () => {
        updateVisibility(value);
        if (onChange) onChange(value);
      });

      const labelEl = document.createElement("label");
      labelEl.setAttribute("for", input.id);
      labelEl.innerText = value;

      container.appendChild(input);
      container.appendChild(labelEl);
      container.appendChild(document.createElement("br"));
    });

    if (Object.entries(visibilityMap).length > 0) {
      const firstInput = document.getElementById(`${Object.entries(visibilityMap)[0][1][0]}-container`);
      this.groups[group].insertBefore(container, firstInput);
    } else {
      this.groups[group].appendChild(container);
    }

    this.io[name] = () => {
      const selected = container.querySelector(`input[name="${name}"]:checked`);
      return selected ? selected.value : null;
    };

    // Initialize visibility state
    updateVisibility(defaultValue);
  }
}
