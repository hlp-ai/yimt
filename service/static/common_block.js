

function set_setting(){
    document.getElementById("url_setting_block").innerHTML = `
        <label id="translation_setting_label">翻译设置</label>
        <div id="translation_setting_block1">
            <label id="server_label">服务器：</label>
            <input type="text" class="url" id="url">
        </div>
        <div id="translation_setting_block2">
            <label id="api_key_label">API KEY：</label>
            <input type="text" class="api_key" id="api_key" placeholder="api_key">
        </div>
        <input type="button" class="url_setting_hide" value="关闭" onclick="url_setting_hide_func()">
        <input type="button" class="url_button" id="url_button" value="设置" onclick="url_resetting_func()">
    `;

    document.getElementById("url_setting_label").addEventListener("click", url_setting_func);
}

function url_setting_func()
{
    document.getElementById("url_setting_block").style.visibility='visible';
    document.getElementById("mask").style.display='block';
    document.getElementById("url").value=END_POINT;
    document.getElementById("api_key").value=API_KEY;
}

function url_resetting_func()
{
    END_POINT = document.getElementById("url").value;
    window.localStorage.setItem("server", END_POINT);

    document.getElementById("url_setting_block").style.visibility='hidden';
    document.getElementById("mask").style.display='none';

    API_KEY = document.getElementById("api_key").value;
    window.localStorage.setItem("key",API_KEY);
}

function url_setting_hide_func()
{
    document.getElementById("url_setting_block").style.visibility='hidden';
    document.getElementById("mask").style.display='none'
}