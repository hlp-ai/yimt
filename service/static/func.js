
async function request_ad(){
    const res = await fetch(END_POINT + "/request_ad", {
        method: "POST",
        body: JSON.stringify({platform:"web"}),
        headers: { "Content-Type": "application/json" },
        }
    );
    trans_json = await res.json();
    ad_id = trans_json.ad_id;
    type = trans_json.type;
    content = trans_json.content;
    url = trans_json.url;
    console.log("ad_id:"+ad_id);
    console.log("ad_type:"+type);
    //console.log("ad_content:"+content);
    //console.log("ad_url:"+url);
    //document.getElementById("content").setAttribute("",)
    if(url != ""){
        ad_url = document.getElementById("product1_url");
        ad_url.style.visibility='visible';
        ad_url.href = url;
        //console.log("ad_url.href: "+ad_url.href);
    }else{
        document.getElementById("product1_url").style.visibility='hidden';
    }
    if(type == "text"){
        document.getElementById("product1_content").style.visibility='visible';
        document.getElementById("product1_content").innerText = content;
        //console.log(content);
    }else if(type == "image"){
        document.getElementById("product1_content").style.visibility='hidden';
        const base64String = content;
        ad_img_div = document.getElementById("product1_area");
        ad_img_div.style.backgroundImage='url('+'data:image/png;base64,'+content+')';
        ad_img_div.style.backgroundSize = "contain";
        ad_img_div.style.backgroundPosition = "center";
        ad_img_div.style.backgroundRepeat = "no-repeat";
    }else{
        document.getElementById("product1_content").innerText = "出错啦";
    }
}