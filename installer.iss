; Inno Setup script - WINDOCSCAN
#define MyAppName "WINDOCSCAN"
#define MyAppVersion "7.6.2"
#define MyAppPublisher "Andrea"
#define MyAppExeName "WINDOCSCAN.exe"
[Setup]
AppId={{C8A4A2B6-31A3-4EAB-8A1B-4E6DDC0A4501}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=installer_output
OutputBaseFilename=WINDOCSCAN_v7_6_2_Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
SetupIconFile=windocscan.ico
[Languages]
Name: "italian"; MessagesFile: "compiler:Languages\Italian.isl"
[Tasks]
Name: "desktopicon"; Description: "Crea un'icona sul desktop"; GroupDescription: "Icone aggiuntive:"; Flags: unchecked
[Files]
Source: "dist\WINDOCSCAN.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "windocscan.ico"; DestDir: "{app}"; Flags: ignoreversion
[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\windocscan.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\windocscan.ico"; Tasks: desktopicon
[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Avvia {#MyAppName}"; Flags: nowait postinstall skipifsilent
