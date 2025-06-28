window.onload = function() {    
    let box = document.getElementsByClassName("box")[0];
    // 或者：let box = document.querySelector(".box");
    box.addEventListener("click", ()=>  {
        alert("这是测试代码");
    });
}