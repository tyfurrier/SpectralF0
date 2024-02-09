

export default function getNames() {
    return fetch('/hello/')
        .then(res => res.json())
        .then(data => {
            return data.content;
        })
}