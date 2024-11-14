import { app } from "../../scripts/app.js"

app.registerExtension({
  name: 'sq.utils.restart',
  async setup() {
    const menu = document.querySelector('.comfy-menu')
    const btn = document.createElement('button')
    btn.innerHTML = 'Restart'
    btn.onclick = () => {
      fetch('/sq/restart')
    }
    menu.append(btn)
  }
})