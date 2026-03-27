#!/usr/bin/env python3
"""
Convierte launch_web_ui.ps1 y setup_ssh_key.ps1 en ejecutables .exe
usando el compilador C# de .NET Framework (csc.exe).
"""

import os
import base64
import subprocess
import sys

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
CSC = r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"

C_TEMPLATE = r"""using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;

[assembly: AssemblyTitle("{title}")]
[assembly: AssemblyDescription("{description}")]
[assembly: AssemblyCompany("GALTTCMC")]
[assembly: AssemblyProduct("{title}")]
[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyCopyright("GALTTCMC 2024")]

class Program {{
    static int Main(string[] args) {{
        string b64 = "{b64}";
        byte[] scriptBytes = Convert.FromBase64String(b64);
        string tmpFile = Path.Combine(
            Path.GetTempPath(),
            "{script_name}_" + Guid.NewGuid().ToString("N").Substring(0, 8) + ".ps1"
        );
        try {{
            File.WriteAllBytes(tmpFile, scriptBytes);
            string arguments = "-ExecutionPolicy Bypass -NoProfile -File \"" + tmpFile + "\"";
            if (args.Length > 0) {{
                arguments += " " + string.Join(" ", args);
            }}
            ProcessStartInfo psi = new ProcessStartInfo("powershell.exe", arguments);
            psi.UseShellExecute = false;
            Process proc = Process.Start(psi);
            proc.WaitForExit();
            return proc.ExitCode;
        }} finally {{
            try {{ if (File.Exists(tmpFile)) File.Delete(tmpFile); }} catch {{ }}
        }}
    }}
}}
"""

scripts = [
    {
        "ps1": "launch_web_ui.ps1",
        "exe": "launch_web_ui.exe",
        "title": "CI/CD Web UI Launcher",
        "description": "Establece tunel SSH y lanza la Web UI del pipeline CI/CD",
    },
    {
        "ps1": "setup_ssh_key.ps1",
        "exe": "setup_ssh_key.exe",
        "title": "Setup SSH Key",
        "description": "Configura autenticacion SSH por clave publica para acceso sin contrasena",
    },
]

ok_count = 0
for s in scripts:
    ps1_path = os.path.join(WORK_DIR, s["ps1"])
    exe_path = os.path.join(WORK_DIR, s["exe"])
    cs_path = os.path.join(WORK_DIR, s["ps1"].replace(".ps1", "_wrapper.cs"))

    print("\n[INFO] Procesando: {}".format(s["ps1"]))

    if not os.path.isfile(ps1_path):
        print("[FAIL] No se encuentra: {}".format(ps1_path))
        continue

    with open(ps1_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")

    script_name = os.path.splitext(s["ps1"])[0]
    cs_code = C_TEMPLATE.format(
        title=s["title"],
        description=s["description"],
        b64=b64,
        script_name=script_name,
    )

    with open(cs_path, "w", encoding="utf-8") as f:
        f.write(cs_code)
    print("[INFO] C# generado: {}".format(cs_path))

    result = subprocess.run(
        [CSC, "/target:exe", "/optimize+", "/out:" + exe_path, cs_path],
        capture_output=True,
        text=True
    )

    os.remove(cs_path)

    if result.returncode == 0:
        size_kb = os.path.getsize(exe_path) // 1024
        print("[OK]  Generado: {} ({} KB)".format(exe_path, size_kb))
        ok_count += 1
    else:
        print("[FAIL] Error compilando {}:".format(s["exe"]))
        print(result.stdout)
        print(result.stderr)

print("\n{}/{} ejecutables generados correctamente.".format(ok_count, len(scripts)))
if ok_count == len(scripts):
    sys.exit(0)
else:
    sys.exit(1)
