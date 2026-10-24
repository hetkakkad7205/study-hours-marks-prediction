
const studyInput = document.getElementById("study_hours");

if (studyInput) {
    studyInput.addEventListener("input", function () {
        const value = Number(this.value);

        if (this.value !== "" && (value < 0 || value > 24)) {
            this.setCustomValidity(
                "Enter study hours between 0 and 24."
            );
        } else {
            this.setCustomValidity("");
        }
    });
}
