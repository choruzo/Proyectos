import { useNavigate } from 'react-router-dom'
import { LoginPage } from './LoginPage'
import { useAuth } from '../auth'

export function LoginRoute() {
  const { login } = useAuth()
  const nav = useNavigate()

  return (
    <LoginPage
      onLoggedIn={async (email, password) => {
        const me = await login(email, password)
        nav(me.is_admin ? '/admin/users' : '/projects', { replace: true })
      }}
    />
  )
}
