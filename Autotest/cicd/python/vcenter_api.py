#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vcenter_api.py - Interacción con vCenter REST API (sin pyvmomi)

Compatible con Python 3.6+
Usa la API REST de vCenter 6.5+ directamente con requests

Uso:
    python3.6 vcenter_api.py <config_path> <action> [args...]
    
Acciones:
    upload_iso <local_iso_path>      - Subir ISO al datastore
    configure_cdrom                  - Configurar CD-ROM de la VM con el ISO
    power_on                         - Encender la VM
    power_off                        - Apagar la VM
    get_vm_status                    - Obtener estado de la VM
    revert_snapshot [snapshot_name]  - Revertir al snapshot (usa config si no se pasa arg)
    wait_powered_off                 - Esperar a que la VM esté apagada
    wait_powered_on                  - Esperar a que la VM esté encendida
"""

from __future__ import print_function
import sys
import os
import yaml
import requests
import json
import time
import urllib3

# Desactivar warnings SSL para entornos con certificados auto-firmados
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class VCenterRESTClient(object):
    """Cliente para vCenter REST API (compatible con Python 3.6)"""
    
    def __init__(self, config):
        self.config = config.get('vcenter', {})
        self.base_url = self.config.get('url', '').rstrip('/')
        self.username = os.environ.get('VCENTER_USER', self.config.get('username', ''))
        self.password = os.environ.get('VCENTER_PASSWORD', self.config.get('password', ''))
        self.session_id = None
        self.verify_ssl = False  # Cambiar a True si tienes certificados válidos
        
    def _get_headers(self):
        """Obtener headers para peticiones autenticadas"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        if self.session_id:
            headers['vmware-api-session-id'] = self.session_id
        return headers
    
    def connect(self):
        """Autenticar y obtener session ID"""
        url = '{}/api/session'.format(self.base_url)
        
        try:
            response = requests.post(
                url,
                auth=(self.username, self.password),
                verify=self.verify_ssl
            )
            response.raise_for_status()
            
            # La respuesta es el session ID como string JSON
            self.session_id = response.json()
            print('[OK] Conectado a vCenter: {}'.format(self.base_url))
            return True
            
        except requests.exceptions.RequestException as e:
            # Intentar con endpoint alternativo (versiones antiguas)
            return self._connect_legacy()
    
    def _connect_legacy(self):
        """Autenticación para vCenter 6.5/6.7 (endpoint antiguo)"""
        url = '{}/rest/com/vmware/cis/session'.format(self.base_url)
        
        try:
            response = requests.post(
                url,
                auth=(self.username, self.password),
                verify=self.verify_ssl
            )
            response.raise_for_status()
            
            data = response.json()
            self.session_id = data.get('value', '')
            print('[OK] Conectado a vCenter (legacy): {}'.format(self.base_url))
            return True
            
        except requests.exceptions.RequestException as e:
            print('[ERROR] No se pudo conectar a vCenter: {}'.format(str(e)))
            raise
    
    def disconnect(self):
        """Cerrar sesión"""
        if not self.session_id:
            return
            
        try:
            url = '{}/api/session'.format(self.base_url)
            requests.delete(url, headers=self._get_headers(), verify=self.verify_ssl)
            print('[OK] Sesión cerrada')
        except Exception:
            pass  # Ignorar errores al cerrar
        finally:
            self.session_id = None
    
    def get_vm(self, vm_name):
        """Obtener VM por nombre"""
        url = '{}/api/vcenter/vm'.format(self.base_url)
        params = {'names': vm_name}
        
        response = requests.get(
            url,
            headers=self._get_headers(),
            params=params,
            verify=self.verify_ssl
        )
        
        if response.status_code == 404:
            # Intentar endpoint legacy
            return self._get_vm_legacy(vm_name)
        
        response.raise_for_status()
        vms = response.json()
        
        if not vms:
            raise Exception('VM no encontrada: {}'.format(vm_name))
        
        return vms[0]
    
    def _get_vm_legacy(self, vm_name):
        """Obtener VM (endpoint legacy)"""
        url = '{}/rest/vcenter/vm'.format(self.base_url)
        params = {'filter.names': vm_name}
        
        response = requests.get(
            url,
            headers=self._get_headers(),
            params=params,
            verify=self.verify_ssl
        )
        response.raise_for_status()
        
        data = response.json()
        vms = data.get('value', [])
        
        if not vms:
            raise Exception('VM no encontrada: {}'.format(vm_name))
        
        return vms[0]
    
    def get_vm_power_state(self, vm_id):
        """Obtener estado de power de la VM"""
        url = '{}/api/vcenter/vm/{}/power'.format(self.base_url, vm_id)
        
        response = requests.get(
            url,
            headers=self._get_headers(),
            verify=self.verify_ssl
        )
        response.raise_for_status()
        
        return response.json().get('state', 'UNKNOWN')
    
    def power_on_vm(self, vm_name=None):
        """Encender la VM"""
        vm_name = vm_name or self.config.get('vm_name')
        vm = self.get_vm(vm_name)
        vm_id = vm.get('vm', vm.get('value', {}).get('vm'))
        
        # Verificar estado actual
        try:
            state = self.get_vm_power_state(vm_id)
            if state == 'POWERED_ON':
                print('[*] VM ya encendida: {}'.format(vm_name))
                return True
        except Exception:
            pass  # Continuar con power on
        
        url = '{}/api/vcenter/vm/{}/power?action=start'.format(self.base_url, vm_id)
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                verify=self.verify_ssl
            )
            response.raise_for_status()
        except Exception:
            # Intentar endpoint legacy
            url = '{}/rest/vcenter/vm/{}/power/start'.format(self.base_url, vm_id)
            response = requests.post(
                url,
                headers=self._get_headers(),
                verify=self.verify_ssl
            )
            response.raise_for_status()
        
        print('[OK] VM encendida: {}'.format(vm_name))
        return True
    
    def power_off_vm(self, vm_name=None):
        """Apagar la VM"""
        vm_name = vm_name or self.config.get('vm_name')
        vm = self.get_vm(vm_name)
        vm_id = vm.get('vm', vm.get('value', {}).get('vm'))
        
        url = '{}/api/vcenter/vm/{}/power?action=stop'.format(self.base_url, vm_id)
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                verify=self.verify_ssl
            )
            response.raise_for_status()
        except Exception:
            # Intentar endpoint legacy
            url = '{}/rest/vcenter/vm/{}/power/stop'.format(self.base_url, vm_id)
            response = requests.post(
                url,
                headers=self._get_headers(),
                verify=self.verify_ssl
            )
            response.raise_for_status()
        
        print('[OK] VM apagada: {}'.format(vm_name))
        return True
    
    def upload_iso_to_datastore(self, local_iso_path, remote_filename=None):
        """
        Subir ISO al datastore vía HTTP PUT
        
        Usa el endpoint de ficheros del datastore:
        PUT https://vcenter/folder/<path>?dcPath=<datacenter>&dsName=<datastore>
        """
        datastore = self.config.get('datastore')
        datacenter = self.config.get('datacenter')
        iso_folder = self.config.get('iso_path', '/ISO').lstrip('/')
        
        if remote_filename is None:
            remote_filename = os.path.basename(local_iso_path)
        
        remote_path = '{}/{}'.format(iso_folder, remote_filename)
        
        # URL para upload via HTTPS
        url = '{}/folder/{}?dcPath={}&dsName={}'.format(
            self.base_url,
            remote_path,
            datacenter,
            datastore
        )
        
        file_size = os.path.getsize(local_iso_path)
        print('[*] Subiendo ISO: {} ({:.2f} GB)'.format(
            local_iso_path, 
            file_size / (1024**3)
        ))
        print('[*] Destino: [{}] {}'.format(datastore, remote_path))
        
        # Subir con streaming para ficheros grandes
        with open(local_iso_path, 'rb') as f:
            response = requests.put(
                url,
                data=f,
                auth=(self.username, self.password),
                verify=self.verify_ssl,
                headers={
                    'Content-Type': 'application/octet-stream',
                    'Content-Length': str(file_size)
                }
            )
        
        if response.status_code in [200, 201, 204]:
            print('[OK] ISO subido: [{}] {}'.format(datastore, remote_path))
            return '[{}] {}'.format(datastore, remote_path)
        else:
            raise Exception('Error subiendo ISO: {} - {}'.format(
                response.status_code, 
                response.text
            ))
    
    def get_vm_hardware(self, vm_id):
        """Obtener configuración de hardware de la VM"""
        url = '{}/api/vcenter/vm/{}/hardware'.format(self.base_url, vm_id)
        
        response = requests.get(
            url,
            headers=self._get_headers(),
            verify=self.verify_ssl
        )
        response.raise_for_status()
        
        return response.json()
    
    def get_cdrom_devices(self, vm_id):
        """Obtener lista de dispositivos CD-ROM de la VM"""
        url = '{}/api/vcenter/vm/{}/hardware/cdrom'.format(self.base_url, vm_id)
        
        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                verify=self.verify_ssl
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            # Endpoint legacy
            url = '{}/rest/vcenter/vm/{}/hardware/cdrom'.format(self.base_url, vm_id)
            response = requests.get(
                url,
                headers=self._get_headers(),
                verify=self.verify_ssl
            )
            response.raise_for_status()
            return response.json().get('value', [])
    
    def configure_cdrom(self, vm_name=None, iso_path=None):
        """
        Configurar CD-ROM de la VM para usar el ISO
        
        iso_path: Ruta en formato [datastore] path/to/file.iso
        """
        vm_name = vm_name or self.config.get('vm_name')
        
        if iso_path is None:
            datastore = self.config.get('datastore')
            iso_folder = self.config.get('iso_path', '/ISO').lstrip('/')
            iso_path = '[{}] {}/{}.iso'.format(datastore, iso_folder, vm_name)
        
        vm = self.get_vm(vm_name)
        vm_id = vm.get('vm', vm.get('value', {}).get('vm'))
        
        # Obtener CD-ROMs de la VM
        cdroms = self.get_cdrom_devices(vm_id)
        
        if not cdroms:
            raise Exception('No se encontró dispositivo CD-ROM en la VM')
        
        # Usar el primer CD-ROM
        cdrom = cdroms[0]
        cdrom_id = cdrom.get('cdrom', cdrom.get('key'))
        
        print('[*] Configurando CD-ROM (id: {})'.format(cdrom_id))
        print('[*] ISO objetivo: {}'.format(iso_path))
        
        # Esperar un momento para que el SO libere el CD-ROM (debe desmontarse antes desde SSH)
        import time
        print('[*] Esperando 2 segundos para que el SO libere el CD-ROM...')
        time.sleep(2)
        
        # PASO 1: Desconectar el CD-ROM (intentar, pero continuar si falla)
        # NOTA: El SO invitado debe haber ejecutado 'eject' antes de llegar aquí
        # para liberar el lock del kernel. Si el SO sigue bloqueando el CD-ROM,
        # vCenter mostrará un diálogo de confirmación que puede colgar el pipeline.
        # El timeout de 30s evita que la llamada se quede bloqueada indefinidamente.
        print('[*] Desconectando CD-ROM...')
        disconnected = False
        try:
            disconnect_url = '{}/rest/vcenter/vm/{}/hardware/cdrom/{}/disconnect'.format(
                self.base_url, vm_id, cdrom_id
            )
            disconnect_response = requests.post(
                disconnect_url,
                headers=self._get_headers(),
                verify=self.verify_ssl,
                timeout=30
            )
            if disconnect_response.status_code in [200, 204]:
                print('[OK] CD-ROM desconectado')
                disconnected = True
                time.sleep(2)  # Esperar a que se aplique la desconexión
            else:
                print('[WARN] Respuesta inesperada al desconectar: {}'.format(disconnect_response.status_code))
        except requests.exceptions.Timeout:
            print('[WARN] Timeout al desconectar CD-ROM (30s). El SO invitado puede seguir bloqueando el dispositivo.')
            print('[*] Asegurese de ejecutar "eject /dev/sr0" en el SO invitado antes de llamar a esta funcion.')
            print('[*] Intentando continuar con el cambio de ISO...')
        except Exception as e:
            print('[WARN] No se pudo desconectar CD-ROM: {}'.format(str(e)))
            print('[*] Intentando continuar con el cambio de ISO...')
        
        # PASO 2: Configurar el backing (cambiar el ISO)
        print('[*] Actualizando backing del CD-ROM...')
        url = '{}/rest/vcenter/vm/{}/hardware/cdrom/{}'.format(
            self.base_url, vm_id, cdrom_id
        )
        
        payload = {
            'spec': {
                'backing': {
                    'type': 'ISO_FILE',
                    'iso_file': iso_path
                },
                'start_connected': True,
                'allow_guest_control': True
            }
        }
        
        try:
            response = requests.patch(
                url,
                headers=self._get_headers(),
                json=payload,
                verify=self.verify_ssl,
                timeout=30
            )
            response.raise_for_status()
            print('[OK] Backing del CD-ROM actualizado')
        except requests.exceptions.Timeout:
            raise Exception('Timeout al actualizar backing del CD-ROM (30s). Verifique que el SO invitado ejecuto "eject" antes.')
        except requests.exceptions.HTTPError as e:
            print('[WARN] API REST falló, intentando con API moderna...')
            # Intentar con API moderna
            url = '{}/api/vcenter/vm/{}/hardware/cdrom/{}'.format(
                self.base_url, vm_id, cdrom_id
            )
            payload = {
                'backing': {
                    'type': 'ISO_FILE',
                    'iso_file': iso_path
                },
                'start_connected': True,
                'allow_guest_control': True
            }
            response = requests.patch(
                url,
                headers=self._get_headers(),
                json=payload,
                verify=self.verify_ssl,
                timeout=30
            )
            response.raise_for_status()
            print('[OK] Backing del CD-ROM actualizado (API moderna)')
        
        time.sleep(2)  # Esperar a que se aplique el cambio
        
        # PASO 3: Conectar el CD-ROM explícitamente
        print('[*] Conectando CD-ROM...')
        max_retries = 3
        connected = False
        
        for attempt in range(1, max_retries + 1):
            try:
                connect_url = '{}/rest/vcenter/vm/{}/hardware/cdrom/{}/connect'.format(
                    self.base_url, vm_id, cdrom_id
                )
                connect_response = requests.post(
                    connect_url,
                    headers=self._get_headers(),
                    verify=self.verify_ssl,
                    timeout=30
                )
                if connect_response.status_code in [200, 204]:
                    print('[OK] CD-ROM conectado (intento {}/{})'.format(attempt, max_retries))
                    connected = True
                    break
                elif connect_response.status_code == 400:
                    # Puede que ya esté conectado
                    print('[*] CD-ROM puede que ya esté conectado (intento {}/{})'.format(attempt, max_retries))
                    connected = True
                    break
                else:
                    print('[WARN] Respuesta inesperada al conectar: {} (intento {}/{})'.format(
                        connect_response.status_code, attempt, max_retries))
                    if attempt < max_retries:
                        time.sleep(2)
            except Exception as e:
                print('[WARN] Error conectando CD-ROM: {} (intento {}/{})'.format(str(e), attempt, max_retries))
                if attempt < max_retries:
                    time.sleep(2)
        
        if not connected:
            print('[ERROR] No se pudo conectar el CD-ROM después de {} intentos'.format(max_retries))
            raise Exception('Fallo al conectar CD-ROM después del cambio de ISO')
        
        print('[OK] CD-ROM configurado correctamente con ISO: {}'.format(iso_path))
        return True
    
    def list_snapshots(self, vm_id):
        """Obtener lista de snapshots de la VM"""
        url = '{}/api/vcenter/vm/{}/snapshot'.format(self.base_url, vm_id)
        try:
            response = requests.get(url, headers=self._get_headers(), verify=self.verify_ssl)
            response.raise_for_status()
            return response.json()
        except Exception:
            url = '{}/rest/vcenter/vm/{}/snapshot'.format(self.base_url, vm_id)
            response = requests.get(url, headers=self._get_headers(), verify=self.verify_ssl)
            response.raise_for_status()
            return response.json().get('value', [])

    def revert_to_snapshot(self, vm_name=None, snapshot_name=None):
        """Revertir la VM al snapshot indicado"""
        vm_name = vm_name or self.config.get('vm_name')
        snapshot_name = snapshot_name or self.config.get('snapshot_name')
        vm = self.get_vm(vm_name)
        vm_id = vm.get('vm', vm.get('value', {}).get('vm'))

        snapshots = self.list_snapshots(vm_id)
        snapshot_id = None
        for s in snapshots:
            if s.get('name') == snapshot_name:
                snapshot_id = s.get('snapshot')
                break

        if not snapshot_id:
            raise Exception('Snapshot no encontrado: {}'.format(snapshot_name))

        url = '{}/api/vcenter/vm/{}/snapshot/{}?action=revert'.format(
            self.base_url, vm_id, snapshot_id)
        try:
            response = requests.post(url, headers=self._get_headers(), verify=self.verify_ssl)
            if response.status_code not in [200, 204]:
                raise Exception('Status {}'.format(response.status_code))
        except Exception:
            url = '{}/rest/vcenter/vm/{}/snapshot/{}/revert'.format(
                self.base_url, vm_id, snapshot_id)
            response = requests.post(url, headers=self._get_headers(), verify=self.verify_ssl)
            response.raise_for_status()

        print('[OK] Snapshot revertido: {}'.format(snapshot_name))
        return True

    def wait_vm_power_state(self, vm_name=None, target_state='POWERED_OFF',
                            max_attempts=30, interval=10):
        """Esperar hasta que la VM alcance el estado de power indicado"""
        vm_name = vm_name or self.config.get('vm_name')
        vm = self.get_vm(vm_name)
        vm_id = vm.get('vm', vm.get('value', {}).get('vm'))

        for attempt in range(1, max_attempts + 1):
            state = self.get_vm_power_state(vm_id)
            if state == target_state:
                print('[OK] VM en estado {}'.format(target_state))
                return True
            print('[*] Esperando {} (actual: {}, intento {}/{})'.format(
                target_state, state, attempt, max_attempts))
            time.sleep(interval)

        raise Exception('Timeout esperando estado {} en VM {}'.format(target_state, vm_name))

    def connect_cdrom(self, vm_name=None):
        """Conectar el CD-ROM (para VMs encendidas)"""
        vm_name = vm_name or self.config.get('vm_name')
        vm = self.get_vm(vm_name)
        vm_id = vm.get('vm', vm.get('value', {}).get('vm'))
        
        cdroms = self.get_cdrom_devices(vm_id)
        if not cdroms:
            raise Exception('No se encontró dispositivo CD-ROM')
        
        cdrom = cdroms[0]
        cdrom_id = cdrom.get('cdrom', cdrom.get('key'))
        
        url = '{}/api/vcenter/vm/{}/hardware/cdrom/{}'.format(
            self.base_url, vm_id, cdrom_id
        )
        
        payload = {'connected': True}
        
        response = requests.post(
            url + '?action=connect',
            headers=self._get_headers(),
            verify=self.verify_ssl
        )
        
        print('[OK] CD-ROM conectado')
        return True


def load_config(config_path):
    """Cargar configuración YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expandir variables de entorno
    def expand_env(obj):
        if isinstance(obj, str):
            if obj.startswith('${') and obj.endswith('}'):
                var_name = obj[2:-1]
                return os.environ.get(var_name, obj)
            return obj
        elif isinstance(obj, dict):
            return {k: expand_env(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_env(i) for i in obj]
        return obj
    
    return expand_env(config)


def main():
    if len(sys.argv) < 3:
        print('Uso: vcenter_api.py <config_path> <action> [args...]')
        print('Acciones:')
        print('  upload_iso <local_iso_path>      - Subir ISO al datastore')
        print('  configure_cdrom [iso_path]       - Configurar CD-ROM de la VM con ISO')
        print('  power_on                         - Encender la VM')
        print('  power_off                        - Apagar la VM')
        print('  get_vm_status                    - Estado de la VM')
        print('  revert_snapshot [snapshot_name]  - Revertir al snapshot')
        print('  wait_powered_off                 - Esperar POWERED_OFF')
        print('  wait_powered_on                  - Esperar POWERED_ON')
        sys.exit(1)
    
    config_path = sys.argv[1]
    action = sys.argv[2]
    
    config = load_config(config_path)
    client = VCenterRESTClient(config)
    
    try:
        client.connect()
        
        if action == 'upload_iso':
            if len(sys.argv) < 4:
                print('Error: Falta ruta del ISO local')
                sys.exit(1)
            local_path = sys.argv[3]
            remote_path = client.upload_iso_to_datastore(local_path)
            # Imprimir el path remoto para que el script padre lo capture
            print('[REMOTE_ISO_PATH] {}'.format(remote_path))
            
        elif action == 'configure_cdrom':
            # Aceptar iso_path opcional como argumento
            iso_path = sys.argv[3] if len(sys.argv) > 3 else None
            client.configure_cdrom(iso_path=iso_path)
            
        elif action == 'power_on':
            client.power_on_vm()
            
        elif action == 'power_off':
            client.power_off_vm()
            
        elif action == 'get_vm_status':
            vm = client.get_vm(config['vcenter']['vm_name'])
            print('VM: {}'.format(json.dumps(vm, indent=2)))

        elif action == 'revert_snapshot':
            snapshot_name = sys.argv[3] if len(sys.argv) > 3 else None
            client.revert_to_snapshot(snapshot_name=snapshot_name)

        elif action == 'wait_powered_off':
            client.wait_vm_power_state(target_state='POWERED_OFF')

        elif action == 'wait_powered_on':
            client.wait_vm_power_state(target_state='POWERED_ON')

        else:
            print('Acción no reconocida: {}'.format(action))
            sys.exit(1)
            
    except Exception as e:
        print('[ERROR] {}'.format(str(e)))
        sys.exit(1)
        
    finally:
        client.disconnect()


if __name__ == '__main__':
    main()
