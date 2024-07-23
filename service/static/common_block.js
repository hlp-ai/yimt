

function display_setting_func()
{
    document.getElementById("settings_block").style.visibility='visible';
    document.getElementById("settings_mask").style.display='block';
    document.getElementById("server_url").value=END_POINT;
    document.getElementById("api_key").value=API_KEY;
}

function reset_setting_func()
{
    END_POINT = document.getElementById("url").value;
    window.localStorage.setItem("server", END_POINT);

    document.getElementById("url_setting_block").style.visibility='hidden';
    document.getElementById("mask").style.display='none';

    API_KEY = document.getElementById("api_key").value;
    window.localStorage.setItem("key",API_KEY);
}

function close_setting_func()
{
    document.getElementById("settings_block").style.visibility='hidden';
    document.getElementById("settings_mask").style.display='none'
}