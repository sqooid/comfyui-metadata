import { api } from "../../scripts/api.js"
import { app } from "../../scripts/app.js"

app.registerExtension({
  name: 'sq.nodes.prompt.display',
  async nodeCreated(node) {
    await new Promise(resolve => setTimeout(resolve, 1000))
    if (node.type === "SQ Prompt Display") {
      const id = node.id.toString()

      const widget = node.widgets.find(obj => obj.name === "actual_prompt")
      widget.inputEl.readOnly = true

      api.addEventListener('sq.prompt.text', (event) => {
        if (event.detail.node === id) {
          widget.inputEl.value = event.detail.text
        }
      })
    }
  }
})